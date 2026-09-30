"""Batch prediction, including unlabeled test images."""
import argparse
from pathlib import Path
from PIL import Image
from fema.data import records
from infer import load,predict


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--backend',choices=['cuda','reference']);p.add_argument('--tile',type=int,default=0)
    a=p.parse_args();model,_=load(a.checkpoint,a.device,a.backend);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for row in records(a.manifest):
        name=row['id']
        if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('id must be a basename')
        pred=predict(model,Image.open(row['image']).convert('RGB'),a.device,a.tile)
        Image.fromarray(pred).save(out/f'{name}.png')


if __name__=='__main__':main()
