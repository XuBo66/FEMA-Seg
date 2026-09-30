"""Aggregate measured evaluation files only; values are never filled from paper."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('results',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    rows=[json.loads(Path(path).read_text()) for path in a.results]
    summary={'inputs':a.results,'n':len(rows),'note':'Confirm same dataset/split/protocol and independent subset seeds before interpreting statistics.'}
    for key in ['mIoU','mPA','mBIoU','GVI_RMSE_fraction']:
        values=[r[key] for r in rows if r.get(key) is not None]
        summary[key]={'values':values,'mean':float(np.mean(values)) if values else None,'sample_std':float(np.std(values,ddof=1)) if len(values)>1 else None}
    Path(a.output).write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
