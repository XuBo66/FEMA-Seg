"""Generate controlled reconstruction experiments; execution is explicit."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
import yaml


def variants(base):
    yield 'full',{}
    yield 'no_frequency',{'model.frequency':False}
    yield 'full_scan',{'model.scan_mode':'full'}
    yield 'no_mgcr',{'model.mgcr':False}
    yield 'bilinear',{'model.eaa':False,'model.sgdu':False}
    yield 'eaa_only',{'model.sgdu':False}
    yield 'sgdu_only',{'model.eaa':False}
    for b in [[12],[9,12],[1,3,5,7]]:yield 'bridge_'+'_'.join(map(str,b)),{'model.bridges':b}
    for seed in [10,42,123,777,999]:yield f'seed_{seed}',{'seed':seed,'subset':f'data/{base["dataset"]}/subset{seed}.json'}
    for tau in [.30,.45,.60]:
        for alpha in [.05,.10,.15]:yield f'tau{tau}_alpha{alpha}',{'model.alpha':alpha,'train_manifest':f'data/{base["dataset"]}/pseudo_tau{tau:.2f}/manifest.jsonl'}
    for offset in [1.,2.,3.]:
        for gamma in [.1,.3,.5]:yield f'offset{offset}_gamma{gamma}',{'model.offset':offset,'train.gamma_static':gamma}
    yield 'sparse_only',{'train.sparse_only':True}
    yield 'boxes_only',{'train_manifest':f'data/{base["dataset"]}/boxes_only/manifest.jsonl'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',default='configs/generated');p.add_argument('--run',action='store_true')
    a=p.parse_args();base=yaml.safe_load(Path(a.base).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for name,changes in variants({'dataset':a.dataset}):
        cfg=copy.deepcopy(base)
        for key,v in changes.items():
            parts=key.split('.');target=cfg
            for part in parts[:-1]:target=target[part]
            target[parts[-1]]=v
        cfg['output']=f'runs/{a.dataset}/{name}';dest=out/f'{a.dataset}_{name}.yaml';dest.write_text(yaml.safe_dump(cfg,sort_keys=False))
        cmd=['python','train.py','--config',str(dest)];jobs.append({'name':name,'command':cmd,'status':'not_run'})
        if a.run:subprocess.run([sys.executable,*cmd[1:]],check=True)
    (out/f'{a.dataset}_jobs.json').write_text(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
