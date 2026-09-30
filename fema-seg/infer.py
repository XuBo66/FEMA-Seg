import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from fema.model import FEMASeg


def load(path,device,backend=None):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    cfg=ck['config'];kwargs=dict(cfg['model'])
    if backend: kwargs['backend']=backend
    model=FEMASeg(**kwargs).to(device);model.load_state_dict(ck['model']);model.eval()
    return model,cfg


@torch.no_grad()
def predict(model,image,device,tile=0):
    x=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()[None].to(device)/255
    x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
    if not tile: return model(x)['logits'].argmax(1)[0].cpu().numpy().astype('uint8')
    h,w=x.shape[-2:];scores=None;count=x.new_zeros(1,1,h,w)
    for top in range(0,h,max(1,tile//2)):
        for left in range(0,w,max(1,tile//2)):
            y=min(top,max(0,h-tile));z=min(left,max(0,w-tile))
            logits=model(x[:,:,y:y+tile,z:z+tile])['logits'].float()
            if scores is None: scores=x.new_zeros(1,logits.shape[1],h,w)
            th,tw=logits.shape[-2:];scores[:,:,y:y+th,z:z+tw]+=logits;count[:,:,y:y+th,z:z+tw]+=1
    return (scores/count).argmax(1)[0].cpu().numpy().astype('uint8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--backend',choices=['reference','cuda']);p.add_argument('--tile',type=int,default=0)
    a=p.parse_args();model,cfg=load(a.checkpoint,a.device,a.backend)
    image=Image.open(a.image).convert('RGB');mask=predict(model,image,a.device,a.tile)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(mask).save(path)
    palette=np.random.default_rng(42).integers(0,256,(256,3),dtype=np.uint8)
    Image.fromarray((.55*np.asarray(image)+.45*palette[mask]).astype('uint8')).save(path.with_suffix('.overlay.png'))
    path.with_suffix('.json').write_text(json.dumps({'GVI_fraction':float(np.isin(mask,cfg['vegetation']).mean()),'tile':a.tile,'checkpoint':a.checkpoint}))


if __name__=='__main__':main()
