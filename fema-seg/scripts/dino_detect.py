"""Adapter to official IDEA-Research/DINO checkout (not Grounding DINO)."""
import argparse
import json
import sys
from pathlib import Path
import torch
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--class-map',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();sys.path.insert(0,str(Path(a.repo).resolve()))
    from main import build_model_main
    from util.slconfig import SLConfig
    from util.utils import clean_state_dict
    import datasets.transforms as T
    cfg=SLConfig.fromfile(a.config);cfg.device=a.device
    model,_,post=build_model_main(cfg)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(clean_state_dict(state['model']),strict=True);model.to(a.device).eval()
    transform=T.Compose([T.RandomResize([800],max_size=1333),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    mapping=json.loads(Path(a.class_map).read_text());src=Path(a.manifest).resolve();result={}
    with torch.no_grad():
        for line in src.read_text().splitlines():
            row=json.loads(line);image=Image.open(src.parent/row['image']).convert('RGB');x,_=transform(image,None)
            pred=post['bbox'](model(x[None].to(a.device)),torch.tensor([[image.height,image.width]],device=a.device))[0]
            result[row['id']]=[{'box_xyxy':box.tolist(),'score':float(s),'class_id':mapping[str(int(c))]}
                               for box,s,c in zip(pred['boxes'].cpu(),pred['scores'].cpu(),pred['labels'].cpu()) if str(int(c)) in mapping]
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result))


if __name__=='__main__':main()
