import json, os, platform
from pathlib import Path
import torch, tokenizers, pytest
runtime = {"python": platform.python_version(), "torch": torch.__version__,
           "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
           "tokenizers": tokenizers.__version__, "pytest": pytest.__version__}
Path(os.environ["H7_RUNTIME_JSON"]).write_text(json.dumps(runtime, sort_keys=True, indent=2) + "\n")
print("H7_RUNTIME", json.dumps(runtime, sort_keys=True), flush=True)
assert runtime["python"] == "3.13.15", runtime
assert runtime["torch"] == "2.11.0+cpu" and runtime["torch_cuda"] is None and not runtime["cuda_available"], runtime
assert runtime["tokenizers"] == "0.22.2" and runtime["pytest"] == "9.0.2", runtime
