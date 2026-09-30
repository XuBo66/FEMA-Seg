import argparse
import json
import time
from pathlib import Path
import torch
from infer import load


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--height',type=int,default=1024);p.add_argument('--width',type=int,default=1024)
    p.add_argument('--repeats',type=int,default=50);p.add_argument('--warmup',type=int,default=10);p.add_argument('--output',default='benchmark.json');p.add_argument('--amp',action='store_true')
    a=p.parse_args()
    if a.repeats<1:raise ValueError('repeats must be positive')
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for paper-style benchmark')
    model,cfg=load(a.checkpoint,'cuda');x=torch.randn(1,3,a.height,a.width,device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=a.amp):
        for _ in range(a.warmup):model(x)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for _ in range(a.repeats):model(x)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
    result={'parameters':sum(p.numel() for p in model.parameters()),'FPS':a.repeats/elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'input':[1,3,a.height,a.width],'amp':a.amp,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'FLOPs':None,'FLOPs_note':'Not claimed: generic profilers omit FFT/selective-scan/grid-sample. Requires explicit operator accounting.'}
    Path(a.output).write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
