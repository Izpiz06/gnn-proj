# code/ — Notebooks and data

**Status: Work in progress.** See the top-level `../README.md` for the
project overview, confirmed blockers, and roadmap.

This folder contains the original notebook-based implementation and its
input data. It is the in-progress state of the project, not a finished
package. The notebooks are being refactored into proper Python modules; until
that is done, treat anything here as experimental.

## Files

| File | Description | Status |
|---|---|---|
| `3050.ipynb` | The main notebook: environment setup, data download, Triton-cache patching, graph construction, model, and a benchmark harness | Runs the data pipeline; the training loop does **not** complete (kernel crash) |
| `t4.ipynb` | Early scratch notebook for a Tesla T4 target | Ignore. It pins a different, incompatible toolchain (torch 2.10.0+cu128, transformers 5.0.0) and contains no implementation |
| `cell_line.mcool` | Hi-C contact matrix (human ESC, hg38) from NCBI GEO GSM5057489 | Real data; **not committed to git** (1.17 GB) |
| `1000genomes_reference.fasta` | Chromosome 22, UCSC hg38 | Real data, but **misleadingly named** — it is not 1000 Genomes data, and it only contains chr22 while `3050.ipynb` builds a whole-genome graph |

## Known blockers (all confirmed)

1. DNABERT-2 segfaults in its Triton flash-attention path on this toolchain
   (annotated in `3050.ipynb` cell with the cache-patching step).
2. Modality fusion `torch.cat([seq_embeddings, h])` fails on a shape
   mismatch (sampled-neighborhood rows vs seed rows).
3. DGL neighbor sampler requires int32 seed node ids; the notebook passes
   int64 (`torch.arange(...)`).
4. Labels are random (`torch.randint`); there is no real prediction task yet.

Details and intended fixes are in the top-level README.

## Toolchain this was developed against

- Python 3.11, PyTorch 2.2.2+cu121, DGL 2.4.0+cu121, transformers 4.35.2,
  bitsandbytes 0.43.3
- NVIDIA GeForce RTX 3050 6 GB Laptop GPU

The preprocessing in `3050.ipynb` rebuilds the graph and re-tokenizes every
50 kb bin from scratch each run; a precomputed/cached representation is a
planned improvement.