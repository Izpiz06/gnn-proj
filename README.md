# Genome Code — Hybrid Sequence + 3D-Genome Deep Learning

A research project that combines **DNA sequence** encoding with **3D chromatin
architecture (Hi-C)** in a single hybrid model. The idea: sequence alone misses
the *spatial* wiring of the genome, and Hi-C alone misses the *linear*
sequence — so fuse both.

This directory contains the **current, in-progress** implementation: the
original Jupyter-notebook code plus its real input data. It is **not** a
finished package yet, and the model does **not** yet train end-to-end (see
[Known issues](#known-issues--blockers)). Everything here is experimental.

---

## Status

> ⚠️ **Work in progress.** The data pipeline works (graph builds, sequences
> tokenize, batches are sampled), but the **training loop does not yet
> complete** — the model currently segfaults / crashes during the first
> backward pass on the configured toolchain. Several real bugs are documented
> in [Known issues](#known-issues--blockers). Expect things to move fast and
> break.

---

## Project plan / approach

The overall goal is a **multimodal genomics model that learns from both the
linear genome and its 3D folding**:

1. **Genome sequence** — 50 kb genomic bins are sliced out of a reference
   genome, tokenized, and embedded with **DNABERT-2-117M**, a BERT-family DNA
   language model. This captures the *sequence semantics* of every bin.
2. **3D genome structure** — a bin-level graph is derived from a **Hi-C
   contact map** at 50 kb resolution: each bin is a node, edges are physical
   contacts between bins, and edge weights are contact frequencies. This
   captures the *spatial* organization of the genome.
3. **Hybrid model (`DNABertGNN`)** — DNABERT embeddings are fed through a
   **2-layer GAT (graph attention network)** that propagates information along
   Hi-C contact edges. The sequence and structural embeddings are concatenated
   and passed to a classification head. The model is trained on a bin-level
   prediction task.
4. **Edge-device benchmarking** — the training harness is VRAM-aware, with
   configurable profiles for small consumer GPUs:
   - `RTX 3050 6 GB` — batch size 2, fanout `[2, 2]`
   - `RTX 4060 8 GB` — batch size 4, fanout `[3, 3]`
   - `Tesla T4 16 GB` — batch size 8, fanout `[4, 4]`

### What's already in place

- ✅ Pinned, working dependency set (deliberately old — DNABERT-2 doesn't
  play nice with modern releases, see [Tech stack](#tech-stack)).
- ✅ Triton flash-attention patch that works around a DNABERT-2 incompatibility.
- ✅ Hi-C (`mcool`) ingestion → DGL graph at 50 kb resolution, with
  self-loops and contact-frequency edge weights.
- ✅ Sequence extraction and tokenization for every genomic bin.
- ✅ DGL 2-layer neighbor-sampler `DataLoader`.
- ✅ Hybrid model definition + fp16 (AMP) training loop with gradient
  checkpointing, gradient accumulation, and 8-bit AdamW for low-VRAM GPUs.
- ✅ Benchmark harness that reports loss, peak VRAM, and throughput.

### Next steps / roadmap

- [ ] Refactor notebooks into proper, importable Python modules.
- [ ] Fix the [fusion shape mismatch](#known-issues--blockers) so the model
      can finish a training step (highest priority).
- [ ] Fix seed-node dtype so the DGL sampler runs cleanly.
- [ ] Replace random labels with a real biological prediction task (e.g.
      bin-level regulatory / epigenomic state classification).
- [ ] Cache preprocessed tokens and graph structure instead of rebuilding
      every run (currently re-tokenizes all bins from scratch each time).
- [ ] Re-tune hyperparameters once training actually completes.
---

## What's in this repo

| File | What it is | Status |
|---|---|---|
| `3050.ipynb` | **The main notebook.** Full pipeline: environment setup, DNABERT-2 Triton patching, data download, Hi-C graph construction, hybrid model, dataloader, and VRAM-profile benchmark harness. | Pipeline works; **training loop crashes** (see blockers). |
| `t4.ipynb` | Early scratch notebook aimed at a Tesla T4. Pins a different, **incompatible** toolchain (torch 2.10.0+cu128, transformers 5.0.0) and has no implementation yet. | Ignore. |
| `cell_line.mcool` | Hi-C contact matrix — human embryonic stem cell (ESC), hg38, from NCBI GEO accession `GSM5057489`. | Real data. **Not committed to git** (~1.17 GB). |
| `1000genomes_reference.fasta` | **Chromosome 22 of the human reference genome (hg38)**, downloaded from UCSC. | Real data. **Not committed to git**. Misleadingly named — it is *not* 1000 Genomes data and only contains chr22. |

> The data files are intentionally gitignored (`*.fasta`, `*.mcool`) because of
> their size. If you clone the repo, re-download them — the notebook cells do
> this automatically.

---

## Tech stack

Everything is pinned to an older, mutually-compatible toolchain because
**DNABERT-2 (April 2023) is incompatible with current PyTorch/Transformers**
(the bundled Triton flash-attention kernel uses an API removed in Triton ≥ 2.1).

| Component | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.2.2+cu121 (ships Triton 2.2.0) |
| CUDA | 12.1 |
| DGL | 2.4.0+cu121 |
| HuggingFace Transformers | 4.35.2 |
| DNABERT-2 | `zhihan1996/DNABERT-2-117M` (remote code) |
| bitsandbytes | 0.43.3 (8-bit AdamW) |
| cooler | latest (Hi-C / `.mcool` reading) |
| Biopython | latest (FASTA parsing) |
| NumPy | < 2 (torch 2.2.x was built against NumPy 1.x) |
| GPU | NVIDIA GeForce RTX 3050 6 GB Laptop (primary dev target) |

### The Triton patch (why it exists)

DNABERT-2's cached `flash_attn_triton.py` calls `tl.dot(a, b, trans_b=True)`,
which Triton **removed in ≥ 2.1**. `3050.ipynb` deletes the stale HF module
cache and rewrites those calls to `tl.dot(a, tl.trans(b))` in every cached copy
of the file. Without this patch, loading DNABERT-2 segfaults the kernel.

---

## Running it

Open `3050.ipynb` in Jupyter **and run the cells in order** — the first cells
install the pinned toolchain and patch Triton, and the notebook explicitly
asks you to **restart the kernel** after the patch step before continuing.

```bash
jupyter notebook 3050.ipynb
```

Dependencies for reference (as installed by the notebook):

```bash
pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torchdata==0.7.1
pip install "numpy<2"
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install transformers==4.35.2 accelerate datasets
pip install cooler matplotlib biopython
pip install "bitsandbytes<0.44" einops
```

Note: `3050.ipynb` sets `os.environ["CPATH"] = "/usr/include/python3.12:..."`
at the top of the main cell. That path was machine-specific to the dev box;
making it configurable instead of hardcoded is on the roadmap.

---

## Known issues & blockers

These are the concrete problems currently standing between the working data
pipeline and a training run that completes:

### 1. DNABERT-2 segfaults on this toolchain without the Triton patch
The cached Triton flash-attention kernel uses the removed `tl.dot(a, b,
trans_b=True)` API. **Workaround:** the patching step in `3050.ipynb`. Note
this is a monkey-patch of a *pinned upstream dependency* — fragile, and
should eventually be replaced (e.g. moving to a non-Triton attention path or
a newer model).

### 2. Modality fusion shape mismatch (`torch.cat`)
The model does `torch.cat([seq_embeddings, h], dim=1)`, where
`seq_embeddings` has one row per **neighborhood (sampled) node** while the GAT
output `h` has one row per **seed/destination node**. The classifier expects
`hidden_dim * 2` per *seed* row. Concrete fix directions: compute BERT
embeddings for the seed rows only, or align/project the two differently-shaped
embeddings before concatenation.

### 3. DGL sampler seed dtype is int64, expects int32
`create_biological_sampler` seeds the dataloader with
`torch.arange(g.num_nodes())`, which defaults to int64; DGL's neighbor sampler
on this build wants int32 seed node ids. Fix:
`torch.arange(g.num_nodes(), dtype=torch.int32)`.

### 4. Labels are random (`torch.randint`)
There is **no real prediction task yet** — node labels are just 0/1 noise, so
anything that trains is not learning biology. A real bin-labeling task is
required before results are meaningful.

### 5. Preprocessing is rebuilt from scratch every run
Graph construction and tokenization of every 50 kb bin happen fresh on each
load. A precomputed/cached representation is a planned improvement.

### 6. The reference FASTA is misaligned with the graph
The `mcool` spans a **whole-genome** contact matrix, and the notebook builds a
whole-genome graph — but the FASTA contains **only chr22**. Bins on other
chromosomes currently fall back to all-`N` sequences (correct but pointless).
Decide whether to restrict the graph to chr22 or fetch a full-genome FASTA.

---

## File layout (planned)

Notebooks are being refactored into a proper package; today the layout is flat:

```
genome_code/
├── 3050.ipynb                  # main notebook (env + patch + data + model + benchmark)
├── t4.ipynb                    # stale T4 scratch, ignore
├── cell_line.mcool             # Hi-C, hESC, hg38 (gitignored, ~1.17 GB)
├── 1000genomes_reference.fasta # chr22 hg38 (gitignored) — misleadingly named
└── README.md
```

Planned structure after the refactor:

```
genome_code/
├── genomecode/
│   ├── data/          # mcool + FASTA ingestion, tokenization, caching
│   ├── model/         # DNABertGNN and friends
│   ├── train/         # training + VRAM profiles + benchmark harness
│   └── patch/         # Triton flash-attn patcher (pinned to DNABERT-2)
├── notebooks/         # thin, demo-oriented notebooks
├── tests/
└── pyproject.toml
```

---

## Contributing / notes

- Keep the pinned toolchain consistent — do **not** casually upgrade
  torch / transformers / triton while DNABERT-2 is in the stack.
- The data files are large and gitignored; don't commit them.
- When working on the fusion fix, prefer small standalone repro cells before
  touching the training loop.
- The `t4.ipynb` toolchain notes (torch 2.10 / transformers 5.0) are stale
  and should not be used as guidance.