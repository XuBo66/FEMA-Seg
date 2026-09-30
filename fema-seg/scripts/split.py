import argparse
import json
import random
import math
from collections import defaultdict
from pathlib import Path


def stratified(rows,fraction,seed):
    groups=defaultdict(list)
    for r in rows:groups[r['stratum']].append(r['id'])
    total=round(len(rows)*fraction)
    counts={k:math.floor(len(v)*fraction) for k,v in groups.items()}
    # Largest remainder allocation: global integer budget, no forced one per stratum.
    order=sorted(groups,key=lambda k:(-(len(groups[k])*fraction-counts[k]),k))
    for k in order[:total-sum(counts.values())]:counts[k]+=1
    rng=random.Random(seed);selected=[]
    for k in sorted(groups):selected+=rng.sample(sorted(groups[k]),counts[k])
    return {'seed':seed,'requested_fraction':fraction,'actual_fraction':len(selected)/len(rows),
            'selected_ids':sorted(selected),'strata_counts':counts,'total':len(rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--fraction',type=float,default=.05)
    a=p.parse_args();rows=[json.loads(s) for s in Path(a.manifest).read_text().splitlines()]
    if not 0<a.fraction<=1:raise ValueError('Invalid fraction')
    result=stratified(rows,a.fraction,a.seed)
    if not result['selected_ids']:raise ValueError('Subset rounded to zero; dataset too small')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
