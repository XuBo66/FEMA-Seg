"""Two-stage single-device training, resumable at epoch boundaries."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from fema.data import records,SegData
from fema.model import FEMASeg
from fema.losses import objective
from fema.metrics import Metrics


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


@torch.no_grad()
def evaluate(model,rows,cfg,device):
    model.eval();m=Metrics(cfg['model']['classes'],cfg['vegetation'])
    for x,y,_,_ in DataLoader(SegData(rows,cfg['model']['classes']),batch_size=1):
        pred=model(x.to(device))['logits'].argmax(1).cpu().numpy()[0]
        m.update(pred,y.numpy()[0])
    return m.result()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--init')
    p.add_argument('--stop-after',type=int,help='Stop after this many total completed epochs; preserves the original schedule')
    args=p.parse_args();cfg=yaml.safe_load(Path(args.config).read_text());t=cfg['train']
    seed_all(cfg['seed']);device=torch.device(cfg['device'])
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    train=records(cfg['train_manifest']);val=records(cfg['val_manifest'])
    if {r['image'] for r in train}&{r['image'] for r in val}: raise ValueError('Train/val leakage')
    selected=set(json.loads(Path(cfg['subset']).read_text())['selected_ids'])
    if not selected.issubset({r['id'] for r in train}) or not selected: raise ValueError('Invalid subset')
    expert=[r for r in train if r['id'] in selected]
    model=FEMASeg(**cfg['model']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=t['lr'],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=t['amp'] and device.type=='cuda') if hasattr(torch,'amp') and hasattr(torch.amp,'GradScaler') else torch.cuda.amp.GradScaler(enabled=t['amp'] and device.type=='cuda')
    start=0;best=-1
    if args.init:
        state=torch.load(args.init,map_location='cpu',weights_only=False)
        model.load_state_dict(state.get('model',state),strict=True)
    hashes={k:hashlib.sha256(Path(cfg[k]).read_bytes()).hexdigest() for k in ['train_manifest','val_manifest','subset']}
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False)
        if ck['config']!=cfg: raise ValueError('Resume config differs')
        if ck['input_sha256']!=hashes: raise ValueError('Resume input manifests/subset changed')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        start=ck['epoch']+1;best=ck['best']
        random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy']);torch.set_rng_state(ck['rng_torch'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(ck['rng_cuda'])
    (out/'resolved.json').write_text(json.dumps({'config':cfg,'input_sha256':hashes,'torch':torch.__version__,'parameters':sum(p.numel() for p in model.parameters()),'initialization':args.init or 'random'},indent=2))
    for epoch in range(start,t['epochs']):
        model.train();stage2=epoch>=t['epochs']-t['fine_epochs']
        warm=t['warmup'];progress=max(0,(epoch-warm)/max(1,t['epochs']-warm-1))
        lr=t['lr']*(epoch+1)/max(1,warm) if epoch<warm else t['min_lr']+(t['lr']-t['min_lr'])*.5*(1+math.cos(math.pi*progress))
        for group in opt.param_groups: group['lr']=lr
        ratio=(epoch-(t['epochs']-t['fine_epochs']))/max(1,t['fine_epochs']-1)
        gamma=t.get('gamma_static',.1+.4*max(0,ratio))
        pseudo=SegData(train,cfg['model']['classes'],t['crop'],True,'pseudo')
        gold=SegData(expert,cfg['model']['classes'],t['crop'],True,'label')
        batch=t['batch'];sparse_only=t.get('sparse_only',False)
        if stage2 and batch%4: raise ValueError('Fine-stage batch must be divisible by 4 for exact 1:3 ratio')
        losses=[]
        for step in range(t.get('steps_per_epoch') or math.ceil(len(train)/batch)):
            samples=[]
            for j in range(batch):
                ds=gold if sparse_only or (stage2 and j<batch//4) else pseudo
                samples.append(ds[random.randrange(len(ds))])
            x,y,w=[torch.stack([s[j] for s in samples]).to(device) for j in range(3)]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=t['amp'] and device.type=='cuda'):
                loss=objective(model(x),y,w,gamma)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
        result=evaluate(model,val,cfg,device)
        score=result['mIoU'];improved=score is not None and score>best
        if improved: best=score
        record={'epoch':epoch,'stage':2 if stage2 else 1,'loss':float(np.mean(losses)),'lr':lr,'gamma':gamma,'validation':result}
        with (out/'history.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        checkpoint={'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'config':cfg,'input_sha256':hashes,
                    'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        torch.save(checkpoint,out/'last.pt')
        if improved: torch.save(checkpoint,out/'best.pt')
        print(json.dumps(record),flush=True)
        if args.stop_after is not None and epoch+1>=args.stop_after: break


if __name__=='__main__': main()
