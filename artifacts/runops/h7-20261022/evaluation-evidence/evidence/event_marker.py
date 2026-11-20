import json, os, sys, time
from pathlib import Path
path = Path(os.environ["H7_EVIDENCE_DIR"]) / (sys.argv[1] + ".started.json")
with path.open("x") as handle:
    json.dump({"event": sys.argv[1], "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, handle)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print("H7_SCIENTIFIC_COMMAND_START", sys.argv[1], flush=True)
