# Morph

This repository is a copy of [hans/nnvb-public](https://github.com/hans/nnvb-public), the code for:

> Jon Gauthier, Canaan Breiss, Matthew Leonard, Edward F. Chang (2025). *Emergent morpho-phonological representations in self-supervised speech models.* EMNLP 2025. [arXiv:2509.22973](https://arxiv.org/abs/2509.22973)

The original code was written for Linux machines with NVIDIA GPUs and has no setup instructions. This copy makes it run on an Apple Silicon Mac (tested on a MacBook Air M4, 32 GB) and documents how to set it up and prepare the data. All credit for the original code goes to its authors; it is used under the MIT License (see `LICENSE`).

## Changes in this copy

| File | Change | Why |
|---|---|---|
| `requirements-mac.txt` | **New.** Package list with tested version pins | The original has no requirements file. Newer `datasets` (3.x+) can't load the repo's LibriSpeech loader, and newer `pyarrow` requires NumPy 2 |
| `apply_mac_edits.py` | **New.** Applies the edits below; safe to run more than once | Also creates `data/pos_counts.pkl`, which is not in git |
| `conf/config.yaml` | `device: cuda` → `device: mps` | Macs have no NVIDIA GPU; `mps` is Apple's GPU |
| `notebooks/analogy/run.ipynb` | `device="cuda"` → `device="cpu"`; removed the cell loading `data/WorldLex_Eng_US.Freq.2.txt` | `mps` doesn't support the 64-bit floats this notebook uses. The WorldLex file isn't distributed with the code, and nothing in the notebook uses it |
| `workflows/librispeech/Snakefile` | `output: directory("data/librispeech_alignments")` | Bug fix: Snakemake rejects a folder output not marked `directory()`, so the alignment download failed |
| `.gitignore` | Also ignores `.venv/` | Keeps the Python environment out of git |

## 1. Setup (once)

### Requirements

- An Apple Silicon Mac (M1–M4)
- [Homebrew](https://brew.sh)
- About 5 GB of free disk for the `dev-clean` data
- An internet connection

### Install

```bash
# system tools
brew install python@3.11 git ffmpeg parallel

# get the code
git clone https://github.com/tanya-jp/Morph.git
cd Morph

# Python environment (must be Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-mac.txt

# fix SSL certificate errors on macOS (makes it permanent for this environment)
echo 'export SSL_CERT_FILE="$(python -m certifi)"' >> .venv/bin/activate
source .venv/bin/activate

# create data/pos_counts.pkl and set the device for your Mac
python apply_mac_edits.py
```

If you use conda, run `conda deactivate` first so that only `.venv` is active.

### Check the install

```bash
python -c "import torch, datasets, snakemake; print(torch.__version__, datasets.__version__, 'MPS:', torch.backends.mps.is_available())"
```

Expected: `datasets` version `2.19.2`, and `MPS: True`.

### Every new Terminal session

```bash
cd Morph
source .venv/bin/activate
```

## 2. Prepare the data

The data steps are run with [Snakemake](https://snakemake.readthedocs.io). You ask for an output, and it runs the steps needed to make it, skipping any whose outputs already exist.

```bash
snakemake -n outputs/preprocessed_data/librispeech-dev-clean    # preview: lists 3 jobs, runs nothing
snakemake -c4 outputs/preprocessed_data/librispeech-dev-clean   # run
```

What happens:

1. **`download_librispeech_raw`** downloads LibriSpeech `dev-clean` (~340 MB) from openslr.org into `data/librispeech/dev-clean/`, and converts the audio from `.flac` to 16 kHz `.wav` with ffmpeg.
2. **`download_librispeech_alignments`** downloads the forced alignments (~620 MB, all splits) from Zenodo into `data/librispeech_alignments/`. These give the start and end time of every word and phone.
3. **`preprocess_librispeech`** runs `notebooks/preprocessing/librispeech.ipynb`, which:
   - matches each recording with its alignments
   - converts British spellings to American
   - removes stress marks from phones
   - groups phones into words, and syllabifies them
   - saves the dataset

The downloads have no progress bar and can take 10–20 minutes. To check progress, run `du -sh .snakemake/storage data` in a second Terminal and see whether the sizes grow.

### Output

- `outputs/preprocessed_data/librispeech-dev-clean/`: the prepared dataset (`data-*.arrow`, `dataset_info.json`, `state.json`)
- `outputs/preprocessing/librispeech-dev-clean/notebook.ipynb`: the executed notebook, with its plots and checks

### Check the data

```bash
python - <<'EOF'
import datasets
ds = datasets.load_from_disk("outputs/preprocessed_data/librispeech-dev-clean")
print("utterances:", len(ds))
item = ds[0]
print("text:", item["text"][:80])
for word, phones in list(zip(item["word_detail"]["utterance"], item["word_phonemic_detail"]))[:5]:
    if word:
        print(f"{word:12s}", [(p["phone"], p["start"], p["stop"]) for p in phones])
EOF
```

Expected output: `utterances: 2703`, then the first words with their phones and start/stop times in audio samples (16,000 per second):

```
utterances: 2703
text: THEY FELT MY PULSE THEY LOOKED AT MY TONGUE THEY LISTENED AT MY CHEST THEY FELT
they         [('DH', 7200, 8320), ('EY', 8320, 9760)]
felt         [('F', 9760, 11040), ('EH', 11040, 12160), ('L', 12160, 12640), ('T', 12640, 13120)]
my           [('M', 13120, 13760), ('AY', 13760, 15360)]
pulse        [('P', 15360, 16960), ('AH', 16960, 19520), ('L', 19520, 20640), ('S', 20640, 22880)]
```

## Troubleshooting

| Error | Fix |
|---|---|
| `CERTIFICATE_VERIFY_FAILED` | Run `export SSL_CERT_FILE="$(python -m certifi)"`, or use the permanent fix in Setup, then re-run. Finished downloads are not repeated |
| `Directory cannot be locked` | `snakemake --unlock` |
| `mkdir: ... File exists` during a download | Delete the half-finished folder (e.g. `data/librispeech/dev-clean`) and re-run |
| `Outputs of incorrect type ... must be flagged with directory()` | The Snakefile fix above is missing; check `workflows/librispeech/Snakefile` |
| Warning about `trust_remote_code` | Harmless |
| GNU parallel citation notice | Harmless; run `parallel --citation` once to silence it |
