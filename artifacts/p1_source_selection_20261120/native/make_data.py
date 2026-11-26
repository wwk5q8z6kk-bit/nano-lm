import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from research.cycle3 import exp_nanoscribe_p1_synthesis_01 as old

root = Path(__file__).resolve().parent
pools = {
    'train': {
        'MEDICATION': 'VEMORIL DANTOVEL NIRASON ZELVIRON CORAVEL BIMATIN',
        'DOSE': '13MG 27MG 43MG 56MG 109MG 218MG',
        'ALLERGY': 'VORALIN SANTOLIN MERICAN TALORIN PENORAN DEVATAL',
        'VITAL_BP': '113/71 121/79 133/85 139/87 117/75 127/81'},
    'dev': {
        'MEDICATION': 'TANOVIL REMASON BIVALOR CELORIN',
        'DOSE': '17MG 34MG 62MG 127MG',
        'ALLERGY': 'VELATOR NAROSEL BENTARIL CARMAVIN',
        'VITAL_BP': '115/73 123/77 137/83 143/89'},
    'test': {
        'MEDICATION': 'RENATOL DOVARIN MILAVON CERTALIN',
        'DOSE': '19MG 38MG 64MG 137MG',
        'ALLERGY': 'VENAROL TEMIRAN CARVILON SOMAVIN',
        'VITAL_BP': '119/74 125/82 135/88 141/90'},
}
seen = set(v for pool in (old.SEEN_VALUES, old.UNSEEN_VALUES) for vs in pool.values() for v in vs)
seen_transcripts = set()
for split,seed,n in [('train',60601,240),('dev',60602,48),('test',60603,96)]:
    pool = {k: tuple(v.split()) for k,v in pools[split].items()}
    vocab = {v for vs in pool.values() for v in vs}
    assert not vocab & seen
    seen.update(vocab)
    with patch.object(old, 'SEEN_VALUES', pool):
        records = old._generate_encounters(np.random.default_rng(seed), n, unseen_mode=False)
    texts = {e['transcript'] for e in records}
    assert len(texts) == n and not seen_transcripts & texts
    seen_transcripts |= texts
    path = root/'data'/f'{split}.jsonl'
    raw = ''.join(json.dumps(e, sort_keys=True) + '\n' for e in records).encode()
    path.write_bytes(raw)
    print(split, seed, n, hashlib.sha256(raw).hexdigest())
