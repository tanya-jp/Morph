"""
Prepare nnvb-public to run on a Mac (no NVIDIA GPU).

Run once from the repo root, with your environment active:
    python apply_mac_edits.py

What it changes (safe to run twice; it skips anything already done):
  1. conf/config.yaml        device: cuda  ->  device: mps   (or cpu on an Intel Mac)
                             Used by wav2vec hidden-state extraction.
  2. notebooks/analogy/run.ipynb   device="cuda"  ->  device="cpu"
                             The analogy step uses 64-bit floats, which Apple's GPU
                             backend (MPS) does not support. It is fast enough on CPU.
  3. notebooks/analogy/run.ipynb   removes the cell that loads
                             data/WorldLex_Eng_US.Freq.2.txt (file not in the repo,
                             and the variable it creates is never used).
  4. data/pos_counts.pkl     creates an empty part-of-speech table. prepare_inputs
                             needs it only for noun/verb ambiguity, which doesn't
                             affect past tense.
  5. workflows/librispeech/Snakefile   marks the alignments output as a folder
                             (directory(...)); without this the download rule fails.
"""

import collections
import json
import pickle
import re
from pathlib import Path

import torch

root = Path.cwd()
if not (root / "Snakefile").exists() or not (root / "src").is_dir():
    raise SystemExit("Run this from the nnvb-public folder (the one containing Snakefile).")

changes = []

# 1. device for hidden-state extraction
device = "mps" if torch.backends.mps.is_available() else "cpu"
cfg = root / "conf/config.yaml"
text = cfg.read_text()
new = re.sub(r"(?m)^device: \w+$", f"device: {device}", text, count=1)
if new != text:
    cfg.write_text(new)
    changes.append(f"conf/config.yaml: device set to {device}")

# 2 + 3. run.ipynb
nb_path = root / "notebooks/analogy/run.ipynb"
nb = json.loads(nb_path.read_text())
before = len(nb["cells"])
nb["cells"] = [c for c in nb["cells"]
               if not (c["cell_type"] == "code" and "WorldLex_Eng_US" in "".join(c["source"]))]
if len(nb["cells"]) < before:
    changes.append("run.ipynb: removed the WorldLex frequency cell")
n_device = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    new_src = [line.replace('device="cuda"', 'device="cpu"') for line in c["source"]]
    n_device += sum(a != b for a, b in zip(c["source"], new_src))
    c["source"] = new_src
if n_device:
    changes.append(f'run.ipynb: device="cuda" -> device="cpu" ({n_device} place(s))')
if len(nb["cells"]) < before or n_device:
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")

# 4. pos_counts stub
pos = root / "data/pos_counts.pkl"
if not pos.exists():
    pos.parent.mkdir(exist_ok=True)
    with open(pos, "wb") as f:
        pickle.dump(collections.defaultdict(dict), f)
    changes.append("data/pos_counts.pkl: created (empty)")

# 5. Snakefile directory() fix
smk = root / "workflows/librispeech/Snakefile"
text = smk.read_text()
new = text.replace('output: "data/librispeech_alignments"',
                   'output: directory("data/librispeech_alignments")')
if new != text:
    smk.write_text(new)
    changes.append("workflows/librispeech/Snakefile: alignments output marked as directory()")

print("\n".join(f"  - {c}" for c in changes) if changes else "Nothing to change; all edits already applied.")
