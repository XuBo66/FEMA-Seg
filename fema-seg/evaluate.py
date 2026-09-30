import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
from fema.data import records
from fema.metrics import Metrics
from infer import load,predict


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--backend',choices=['reference','cuda']);p.add_argument('--tile',type=int,default=0)
    a=p.parse_args();model,cfg=load(a.checkpoint,a.device,a.backend);m=Metrics(cfg['model']['classes'],cfg['vegetation'])
    for row in records(a.manifest):
        pred=predict(model,Image.open(row['image']).convert('RGB'),a.device,a.tile)
        target=np.array(Image.open(row['label']),dtype=np.int64);m.update(pred,target)
    result=m.result();result.update(manifest=str(Path(a.manifest).resolve()),checkpoint=a.checkpoint,tile=a.tile)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
