"""Nano LM deterministic sanity examples. Conditional checks, NOT formal verification or LM training.
Read-only by default; use --out NEW.json to save without overwriting frozen records.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np

def check_versioned():
    ev=[("a","old",1,"e1"),("a","new",4,"e2"),("a",None,7,"e3")]
    def latest(es,t):
        valid=[e for e in es if e[0]=="a" and e[2]<=t]
        return max(valid,key=lambda e:(e[2],e[3]))[1] if valid else None
    for es in itertools.permutations(ev):
        assert latest(es,3)=="old" and latest(es,5)=="new" and latest(es,8) is None
    return {"test":"versioned_correctness","passed":True}

def check_transport():
    rng=np.random.default_rng(1001);d=8
    z=rng.normal(size=d);w=rng.normal(size=(4,d))
    q,_=np.linalg.qr(rng.normal(size=(d,d)))
    diff=float(np.max(np.abs((w@z)-(w@q.T)@(q@z))))
    assert diff<1e-12
    return {"test":"known_invertible_transport","max_abs_logit_error":diff,"passed":True}

def check_projection():
    rng=np.random.default_rng(2002)
    j=rng.normal(size=(3,8));g=rng.normal(size=8)
    delta=-(np.eye(8)-np.linalg.pinv(j)@j)@g*.03
    residual=float(np.linalg.norm(j@delta))
    assert residual<1e-12
    return {"test":"first_order_nullspace","linear_residual":residual,"passed":True}

def check_storage():
    assert math.ceil(4*math.log2(8))==12
    return {"test":"independent_storage_bound","minimum_bits":12,"passed":True}

def check_myopic():
    assert .14>.1+.02 and not (.09>.1+.02)
    return {"test":"one_step_compute_rule","passed":True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",help="Optional NEW JSON result; no overwrite")
    args=ap.parse_args()
    rows=[fn() for fn in (check_versioned,check_transport,check_projection,check_storage,check_myopic)]
    result={"n_tests":len(rows),"n_passed":len(rows),"scope":"Deterministic numerical examples only; no LM superiority proof","tests":rows}
    if args.out:
        with Path(args.out).open("x",encoding="utf8") as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
