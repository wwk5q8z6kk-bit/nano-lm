# Sample from the trained checkpoint (temperature + top-p per vault token-sampling page)
import sys, numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer
from train import GPT, dev, V, S

m = GPT().to(dev); m.load_state_dict(torch.load("ckpt.pt", map_location=dev)); m.eval()
tok = Tokenizer.from_file("tokenizer.json")
prompt = sys.argv[1] if len(sys.argv) > 1 else "The history of"
ids = tok.encode(prompt).ids
with torch.no_grad():
    for _ in range(120):
        x = torch.tensor(ids[-S:], device=dev)[None]
        pad = S - x.shape[1]
        if pad > 0: x = F.pad(x, (0, pad))          # model expects fixed S; mask via slice
        logits = m(x)[0, len(ids[-S:]) - 1] / 0.8    # temperature 0.8
        probs = F.softmax(logits, -1)
        sp, si = probs.sort(descending=True)
        keep = (sp.cumsum(0) <= 0.95); keep[0] = True  # top-p 0.95
        idx = si[keep][torch.multinomial(sp[keep] / sp[keep].sum(), 1)]
        ids.append(idx.item())
print(tok.decode(ids))
