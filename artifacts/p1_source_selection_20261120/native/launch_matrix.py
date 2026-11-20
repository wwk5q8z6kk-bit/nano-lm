"""Perform all nine trainings with three CPU workers. Optional output transport via job logging."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import os
from pathlib import Path
import subprocess
import sys

def work(seed, root, stream):
    for arm in ('full','answer','aligned'):
        tag=f'{arm}-{seed}'
        env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable,'-B','run.py',str(seed),arm,str(root)],check=True,env=env)
        for suffix in ('json','pt'):
            path=root/f'{tag}.{suffix}'
            raw=path.read_bytes()
            if stream:
                encoded=base64.b64encode(raw).decode()
                parts=[encoded[i:i+10000] for i in range(0,len(encoded),10000)]
                for i,part in enumerate(parts):
                    print(f'NANOFILE {path.name} {i} {len(parts)} {hashlib.sha256(raw).hexdigest()} {part}',flush=True)
        print('NANOFINISHED '+tag,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('--log-files',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(work,s,a.output,a.log_files) for s in (1481,3527,6029)]
        for future in futures: future.result()
