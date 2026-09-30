import math
import torch
from torch import nn
from torch.nn import functional as F


class Norm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class BandSplit(nn.Module):
    """Smooth radial bands; independent real/imag processing, real IFFT output.

    Complex parameterization and initial radii are reconstruction choices.
    """
    def __init__(self, c):
        super().__init__()
        self.radii = nn.Parameter(torch.tensor([-1.1, 0.0]))
        self.filters = nn.ModuleList([nn.Sequential(
            nn.Conv2d(2*c, 2*c, 3, padding=1, groups=c),
            nn.BatchNorm2d(2*c)) for _ in range(3)])
        self.local = nn.Sequential(nn.Conv2d(c,c,3,padding=1,groups=c), nn.Conv2d(c,c,1))

    def forward(self, x):
        h, w = x.shape[-2:]
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = torch.fft.fftshift(torch.fft.fft2(x.float(), norm='ortho'), dim=(-2,-1))
            fy = torch.fft.fftshift(torch.fft.fftfreq(h, device=x.device)) * 2
            fx = torch.fft.fftshift(torch.fft.fftfreq(w, device=x.device)) * 2
            distance = (fy[:,None].square()+fx[None,:].square()).sqrt()
            r1 = self.radii[0].sigmoid()
            r2 = r1 + (1-r1)*self.radii[1].sigmoid()
            lo, hi = torch.sigmoid((r1-distance)*20), torch.sigmoid((distance-r2)*20)
            # Nested sigmoids give a nonnegative partition of unity.
            masks = [lo, 1-lo-hi, hi]
            merged = 0
            for mask, layer in zip(masks, self.filters):
                q = layer(torch.cat([(z*mask).real, (z*mask).imag], 1))
                re, im = q.chunk(2,1)
                merged = merged + torch.complex(re, im)
            spectral = torch.fft.ifft2(torch.fft.ifftshift(merged,dim=(-2,-1)),norm='ortho').real
        return F.leaky_relu(self.local(x)+spectral.to(x.dtype), .1)


def scan_reference(u, delta, A, B, C, D):
    """Mamba-style selective recurrence. CPU verification only, O(L) Python loop."""
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    ys = []
    for t in range(u.shape[-1]):
        dt = delta[:,:,t,None]
        state = torch.exp(dt*A[None])*state + dt*B[:,None,:,t]*u[:,:,t,None]
        ys.append((state*C[:,None,:,t]).sum(-1)+D*u[:,:,t])
    return torch.stack(ys,-1)


class SelectiveSSM(nn.Module):
    def __init__(self, c, state, backend):
        super().__init__()
        self.backend = backend
        self.params = nn.Conv1d(c, c+2*state, 1)
        self.logA = nn.Parameter(torch.arange(1,state+1).float().log().repeat(c,1))
        self.D = nn.Parameter(torch.ones(c))
        self.c, self.state = c, state

    def forward(self, u):
        dtype = u.dtype
        with torch.autocast(device_type=u.device.type, enabled=False):
            u = u.float()
            dt, b, c = torch.split(self.params(u), [self.c,self.state,self.state],1)
            dt = F.softplus(dt-3)
            A = -self.logA.exp()
            if self.backend == 'cuda':
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                out = selective_scan_fn(u.contiguous(),dt.contiguous(),A,b.contiguous(),c.contiguous(),self.D)
            elif self.backend == 'reference':
                out = scan_reference(u,dt,A,b,c,self.D)
            else:
                raise ValueError('backend must be cuda or reference')
        return out.to(dtype)


class BDSS(nn.Module):
    def __init__(self,c,state,parity,backend,mode='parity'):
        super().__init__()
        self.parity, self.mode = parity, mode
        self.norm = Norm(c)
        self.dw = nn.Conv2d(c,c,3,padding=1,groups=c)
        self.ssm = SelectiveSSM(c,state,backend)
        self.out = nn.Conv2d(c,c,1)
        self.ffn = nn.Sequential(Norm(c), nn.Conv2d(c,4*c,1),nn.GELU(),nn.Conv2d(4*c,c,1))

    def forward(self,x):
        v = F.silu(self.dw(self.norm(x)))
        b,c,h,w = v.shape
        grid = torch.arange(h*w,device=x.device).reshape(h,w)
        selected = ((grid//w+grid%w)%2 == self.parity) if self.mode == 'parity' else torch.ones_like(grid,dtype=torch.bool)
        row, col = grid[selected], grid.T[selected.T]
        flat = v.flatten(2)
        merged = torch.zeros_like(flat)
        # Four directions on half-grid = 2HW token visits.
        for ids in (row,row.flip(0),col,col.flip(0)):
            if ids.numel():
                seq = self.ssm(flat.index_select(2,ids))
                merged = merged.index_add(2,ids,seq/4)
        x = x+self.out(merged.reshape(b,c,h,w))
        return x+self.ffn(x)


def position(c,h,w,device):
    assert c%4 == 0
    freq = torch.exp(-math.log(10000)*torch.arange(c//4,device=device)/(c//4))
    y = torch.arange(h,device=device)[:,None]*freq
    x = torch.arange(w,device=device)[:,None]*freq
    y = torch.cat([y.sin(),y.cos()],-1).T[:,:,None].expand(-1,-1,w)
    x = torch.cat([x.sin(),x.cos()],-1).T[:,None,:].expand(-1,h,-1)
    return torch.cat([y,x],0)[None]


class EVT(nn.Module):
    def __init__(self,c,heads):
        super().__init__()
        self.n1,self.n2 = nn.LayerNorm(c),nn.LayerNorm(c)
        self.compress = nn.Linear(9*c,c)
        self.q,self.k,self.v,self.proj = [nn.Linear(c,c) for _ in range(4)]
        self.ff = nn.Sequential(nn.Linear(c,4*c),nn.GELU(),nn.Linear(4*c,c))
        self.heads = heads

    def forward(self,x,stride):
        b,c,h,w=x.shape
        tokens=x.flatten(2).transpose(1,2)
        norm=self.n1(tokens)
        compressed=self.compress(F.unfold(norm.transpose(1,2).reshape(b,c,h,w),3,padding=1,stride=stride).transpose(1,2))
        def heads(t):
            return t.reshape(b,-1,self.heads,c//self.heads).transpose(1,2)
        q,k,v=heads(self.q(norm)),heads(self.k(compressed)),heads(self.v(compressed))
        att=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,h*w,c)
        tokens=tokens+self.proj(att)
        tokens=tokens+self.ff(self.n2(tokens))
        return tokens.transpose(1,2).reshape(b,c,h,w)


class MGCR(nn.Module):
    def __init__(self,dims,c,depth,heads,bridges,strides):
        super().__init__()
        self.proj=nn.ModuleList([nn.Conv2d(d,c,1) for d in dims])
        self.units=nn.ModuleList([EVT(c,heads) for _ in range(depth)])
        self.norms=nn.ModuleDict({str(i):Norm(c) for i in bridges})
        self.bridges,self.strides=bridges,strides

    def forward(self,features):
        result=[]
        for f,p,s in zip(features,self.proj,self.strides):
            x=p(f); x=x+position(*x.shape[1:],x.device).to(x.dtype)
            bridge=0
            for i,unit in enumerate(self.units,1):
                x=unit(x,s)
                if i in self.bridges:
                    bridge=bridge+self.norms[str(i)](x)
            result.append(bridge)
        return result


class SGDU(nn.Module):
    def __init__(self,c,offset):
        super().__init__()
        self.up=nn.Sequential(nn.Conv2d(c,4*c,1),nn.PixelShuffle(2))
        self.shift=nn.Conv2d(c,2,1)
        nn.init.zeros_(self.shift.weight);nn.init.zeros_(self.shift.bias)
        self.offset=offset

    def forward(self,x):
        y=self.up(x); b,c,h,w=y.shape
        yy,xx=torch.meshgrid(torch.arange(h,device=x.device),torch.arange(w,device=x.device),indexing='ij')
        grid=torch.stack([(xx+.5)*2/w-1,(yy+.5)*2/h-1],-1).float()[None]
        shift=self.shift(y).float().tanh().permute(0,2,3,1)*self.offset
        shift=shift*shift.new_tensor([2/w,2/h])
        return F.grid_sample(y.float(),grid+shift,padding_mode='border',align_corners=False).to(y.dtype)


class Decode(nn.Module):
    def __init__(self,c,classes,alpha,offset,eaa,sgdu):
        super().__init__()
        self.a=nn.Sequential(nn.Conv2d(c,c,1),nn.BatchNorm2d(c),nn.Sigmoid())
        self.b=nn.Sequential(nn.Conv2d(c,c,1),nn.BatchNorm2d(c),nn.Sigmoid())
        self.fuse=nn.Conv2d(2*c,c,1)
        self.cls=nn.Conv2d(c,classes,1)
        self.refine=nn.Sequential(nn.Conv1d(c,c,1),nn.GELU(),nn.Conv1d(c,c,1))
        self.edge=nn.Conv1d(c,1,1)
        self.alpha,self.eaa=alpha,eaa
        self.up=SGDU(c,offset) if sgdu else nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)

    def forward(self,deep,local):
        deep=F.interpolate(deep,size=local.shape[-2:],mode='bilinear',align_corners=False)
        f=self.fuse(torch.cat([local*(1+self.a(deep)),deep*(1+self.b(local))],1))
        logits=self.cls(f)
        pair=logits.float().softmax(1).topk(2,dim=1).values
        uncertainty=1-pair[:,0]+pair[:,1]
        aux={'logits':logits,'uncertainty':uncertainty}
        if self.eaa and self.alpha>0:
            k=max(1,round(f.shape[-1]*f.shape[-2]*self.alpha))
            ids=uncertainty.flatten(1).topk(k,dim=1).indices
            selected=f.flatten(2).gather(2,ids[:,None].expand(-1,f.shape[1],-1))
            refined=self.refine(selected)
            lifted=torch.zeros_like(f.flatten(2)).scatter(2,ids[:,None].expand_as(refined),refined)
            f=f+lifted.reshape_as(f)
            aux.update(indices=ids,edge=self.edge(refined).squeeze(1))
        return self.up(f),aux


class FEMASeg(nn.Module):
    def __init__(self,classes=19,dims=(64,128,256,512),depths=(2,2,4,2),state=16,
                 context_dim=128,context_depth=12,heads=4,bridges=(3,6,9,12),
                 strides=(8,4,2,1),alpha=.1,offset=2.,backend='cuda',
                 frequency=True,mgcr=True,eaa=True,sgdu=True,scan_mode='parity'):
        super().__init__()
        assert len(dims)==len(depths)==len(strides)==4 and classes>=2
        assert 0<=alpha<=1 and bridges and max(bridges)<=context_depth
        self.down,self.stages=nn.ModuleList(),nn.ModuleList()
        prev,layer=3,0
        for i,(c,d) in enumerate(zip(dims,depths)):
            self.down.append(nn.Conv2d(prev,c,4 if i==0 else 2,stride=4 if i==0 else 2))
            blocks=[]
            for _ in range(d):
                if frequency: blocks.append(BandSplit(c))
                blocks.append(BDSS(c,state,layer%2,backend,scan_mode));layer+=1
            self.stages.append(nn.Sequential(*blocks));prev=c
        self.context=MGCR(dims,context_dim,context_depth,heads,bridges,strides) if mgcr else None
        self.local=nn.ModuleList([nn.Conv2d(d,context_dim,1) for d in dims])
        self.decoder=nn.ModuleList([Decode(context_dim,classes,alpha,offset,eaa,sgdu) for _ in dims])
        self.head=nn.Conv2d(context_dim,classes,1)

    def forward(self,x):
        size=x.shape[-2:]
        x=F.pad(x,(0,(-size[1])%32,0,(-size[0])%32))
        padded=x.shape[-2:];features=[]
        for down,stage in zip(self.down,self.stages):
            x=stage(down(x));features.append(x)
        local=[p(f) for p,f in zip(self.local,features)]
        contexts=self.context(features) if self.context else local
        x=contexts[-1];aux=[]
        for i in range(3,-1,-1):
            x,a=self.decoder[i](x,local[i]+contexts[i]);aux.append(a)
        logits=F.interpolate(self.head(x),size=padded,mode='bilinear',align_corners=False)
        return {'logits':logits[:,:,:size[0],:size[1]],'aux':aux,'padded_size':padded}
