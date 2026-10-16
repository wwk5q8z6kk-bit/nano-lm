# Sample from the trained checkpoint (temperature + top-p per vault token-sampling page)
import sys, torch, torch.nn.functional as F
from tokenizers import Tokenizer
import importlib.util
spec = importlib.util.spec_from_file_location("train", "train.py")
# import model classes without re-running training: read file, exec only class defs
src = open("train.py").read().split("m = GPT()")[0]
ns = {}
exec(src, ns)
GPT, dev, V, S = ns["GPT"], ns["dev"], ns["V"], ns["S"]

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
