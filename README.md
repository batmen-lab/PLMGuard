# PLMGuard

PLMGuard is a diagnostic framework for protein sequence search that probes whether similarity scores are biologically meaningful, semantically coherent, and resistant to manipulation. It helps distinguish trustworthy search signals from opaque embedding-based similarity across six complementary experiments.

![Overview](figs/fig_overview.png)


---

## Usage

### Setup

#### 1. Organize the project directory

Download the required assets (links below) and arrange the project as follows:

```
PLMGuard/
├── data/
│   ├── db/                               # Reference databases
│   │   ├── astral.fa                     # ASTRAL sequence database
│   │   └── astral_pdb/                   # ASTRAL PDB structures
│   │       └── astral40/pdbstyle-2.08/
│   └── rosetta_mut/                      # Rosetta-relaxed mutant PDB structures (Exp. 2)
├── libs/                                 # Third-party tool source code
├── results/                              # Output figures and metrics (auto-created)
└── src/                                  # Source code (this repository)
```

- **Sequence database** — [LINK]
- **Library dependencies (PLM methods)** — [LINK]
- **Rosetta-mutated structures** (required for Experiment 2) — [LINK]
- **PDB structures** - [pdbstyle-2.08](https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz)

#### 2. Configure environment paths

Edit `src/PLMs_cmds/.env` and set `BASE_DIR` to your PLMGuard root:

```bash
# src/PLMs_cmds/.env
export BASE_DIR="/path/to/PLMGuard"   # <-- update this line
export DATA_DIR="$BASE_DIR/data"
export TEMP_DIR="$DATA_DIR/temp"
```

The remaining variables in `.env` (Python paths, GPU assignments, library paths) should be updated to match your installation of each tool.

#### 3. Set up conda environments

Create the base evaluation environment:

```bash
conda create -n PLMGuard python=3.10
conda activate PLMGuard
pip install -r requirements.txt
```

Each PLM-based search tool requires its own environment. Follow the per-tool instructions in the linked library README. The tool-specific Python executables are set via `*_PYTHON_PATH` variables in `.env`.

Additionally, build **TMscore** and place the binary at `libs/TMscore`:

```bash
wget https://zhanggroup.org/TM-score/TMscore.cpp -O libs/TMscore.cpp
g++ -static -O3 -ffast-math -lm -o libs/TMscore libs/TMscore.cpp
```

#### 4. Set up search methods

**PLM-based methods** — download our modified versions [LINK], place in `libs/`, and follow the per-tool environment setup:

| Method | Environment variable |
|---|---|
| DCTdomain | `DCT_DOMAIN_PYTHON_PATH`, `DCT_DOMAIN_SRC_DIR` |
| DHR | `DHR_PYTHON_PATH`, `DHR_SRC_DIR` |
| PLMSearch | `PLMSEARCH_PYTHON_PATH`, `PLMSEARCH_SRC_DIR` |
| TM-Vec | `TMVEC_PYTHON_PATH`, `TMVEC_SRC_DIR` |

**Alignment-based methods** — install from official sources at the versions used in the paper:

| Method | Version | Source |
|---|---|---|
| BLASTp | 2.17.0 | https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/ |
| MMseqs2 | 18.8cc5cs | https://github.com/soedinglab/MMseqs2 |
| Diamond | 2.1.14 | https://github.com/bbuchfink/diamond |
| NEAR | — | https://github.com/TravisWheelerLab/NEAR |

---

## Reproduce figures in the paper

Since running all search methods is time-consuming, we provide precomputed parsed search results so you can reproduce all figures directly.

1. Download the precomputed parsed results from the [PLMGuard dataset on Hugging Face](https://huggingface.co/datasets/Hanhanhanhaner/PLMGuard) and place the contents in `data/`
2. Run from the `src/` directory:

```bash
cd src

python sanity_check/check_bio_evolution.py        # Experiment 1
python sanity_check/check_bio_mut_structure.py    # Experiment 2
python sanity_check/check_pert_doublen.py         # Experiment 3
python sanity_check/check_pert_truncation.py      # Experiment 4
python sanity_check/check_perm_data.py            # Experiment 5
python sanity_check/check_perm_model.py           # Experiment 6

python sanity_check/overall_performance.py        # Overall summary bubble plot
```

All figures and metrics are saved under `results/`.

---

## Test pipeline

> **Note:** This repository is designed to test the methods evaluated in the paper. A standalone tool for testing custom search methods is coming soon.

### Prepare sequence database variants

Generate all FASTA variants required by the six experiments from the base `astral.fa`. Run from `src/`:

```bash
cd src

# Evolutionary mutants — Experiment 1 (produces astral_mutantWAG.fa)
python utils/fasta_utils.py --fasta_url ../data/db/astral.fa --type mutant

# Doubled sequences — Experiment 3 (produces astral_doubshuf.fa and astral_doubself.fa)
python utils/fasta_utils.py --fasta_url ../data/db/astral.fa --type doublen

# Truncated sequences — Experiment 4 (produces astral_trunchalf.fa and astral_truncqrt.fa)
python utils/fasta_utils.py --fasta_url ../data/db/astral.fa --type trunclen

# Shuffled database — Experiment 5
# Step 1: generate shuffled sequences (produces astral_shuf.fa)
python utils/fasta_utils.py --fasta_url ../data/db/astral.fa --type shuf
# Step 2: concatenate with originals to produce the decoy database used in Exp. 5
cat ../data/db/astral_shuf.fa ../data/db/astral.fa > ../data/db/astral_shuf_ori.fa

# Convert all FASTA files to TSV (required by DHR)
for fa in ../data/db/astral*.fa; do
    python utils/fasta_utils.py --fasta_url "$fa" --type to_tsv
done
```

This produces files in `data/db/`: `astral_mutantWAG.fa`, `astral_doubshuf.fa`, `astral_doubself.fa`, `astral_trunchalf.fa`, `astral_truncqrt.fa`, `astral_shuf.fa`, and corresponding `.tsv` files.

---

### Search command reference

All `run_*.sh` scripts accept these flags:

| Flag | Description |
|---|---|
| `-q` | Query file (`.fa` for most methods; `.tsv` for DHR) |
| `-t` | Target database file (`.fa` for most methods; `.tsv` for DHR) |
| `-k` | `DB_KHITS`: number of hits the search tool returns |
| `-p` | `PARSER_KHITS`: number of hits the parser keeps; determines the `_hitN` suffix in parsed result filenames |

Since DCTdomain divides each target sequence into multiple segments, the effective number of targets increases. Therefore, the `-k` parameter should be scaled accordingly, i.e., `-k` = 3 × `-p`.

Per-experiment `-k`/`-p` values (driven by target database size and analysis requirements):

| Experiment | Methods | `-p` | `-k` (DCTdomain) | `-k` (others) |
|---|---|---|---|---|
| 1 — Evolutionary Plausibility | all 8 | 1000 | 3000 | 1000 |
| 2 — Structure Consistency | all 8 | 15177 | 45531 | 15177 |
| 3 — Redundancy Stability | all 8 | 1000 | 3000 | 1000 |
| 4 — Similarity Monotonicity | all 8 | 1000 | 3000 | 1000 |
| 5 — Decoy Sensitivity | blastp, near, plm, tmvec, dctdomain, dhr | 30354 | 91062 | 30354 |
| 6 — Representation Reliability (perm only) | dctdomain, dhr, plm, tmvec | 15177 | 45531 | 15177 |

---

### Experiment 1 — Evolutionary Plausibility

Tests whether similarity scores correctly track evolutionary distance across successive stages of amino-acid substitution.

**Run search methods** (all 8 methods; query = original or mutant, target = original database):

```bash
cd src
source PLMs_cmds/.env

# Run for each query in {astral.fa, astral_mutantWAG.fa}; target: astral.fa
# -k/-p: see table above (dctdomain: -k 3000 -p 1000; all others: -k 1000 -p 1000)
bash PLMs_cmds/run_blastp.sh -q ../data/db/astral.fa  -t ../data/db/astral.fa  -k 1000 -p 1000
# ... repeat for diamond, mmseq2, near, plm, tmvec, dctdomain
bash PLMs_cmds/run_dhr.sh    -q ../data/db/astral.tsv -t ../data/db/astral.tsv -k 1000 -p 1000
```

**Run sanity check:**

```bash
python sanity_check/check_bio_evolution.py
```

Output saved to `results/check_bio_evolution/`.

---

### Experiment 2 — Structure Consistency

Tests whether similarity scores reflect the structural relationships of mutant proteins, a task that is especially challenging in the remote-homology regime where sequence identity provides limited guidance.

**Step 1 — Generate Rosetta mutants:**

```bash
cd src
python utils/run_mut_rosetta.py
```

This samples 1000 sequences from `data/db/astral.fa`, runs Rosetta mutation and side-chain relaxation for each, and writes the mutated PDB files to `data/rosetta_mut/`.

Since structure mutation is extremely time-costing, we won't suggest running this script, instead, using our generated data under `data/rosetta_mut`.

**Step 2 — Filter mutants and extract origin sequences:**

```bash
python utils/filter_mut_rosetta.py \
    --data-dir ../data/rosetta_mut
```

This scores each model with PyRosetta (CA-RMSD, fa_rep, backbone torsion, H-bonds) and produces:
- `data/rosetta_mut/astral_mutrosetta.fa` — quality-filtered mutant sequences (CA-RMSD < 3 Å)
- `data/rosetta_mut/astral_originrosetta.fa` — corresponding original sequences extracted from `astral.fa`

Copy both FASTA files to the database directory and convert to TSV for DHR:

```bash
cp ../data/rosetta_mut/astral_mutrosetta.fa    ../data/db/
cp ../data/rosetta_mut/astral_originrosetta.fa ../data/db/
python utils/fasta_utils.py --fasta_url ../data/db/astral_mutrosetta.fa    --type to_tsv
python utils/fasta_utils.py --fasta_url ../data/db/astral_originrosetta.fa --type to_tsv
```

**Step 3 — Run search methods:**

```bash
source PLMs_cmds/.env

# Run for each query in {astral_originrosetta.fa, astral_mutrosetta.fa}; target: astral.fa
# -k/-p: see table above (dctdomain: -k 45531 -p 15177; all others: -k 15177 -p 15177)
bash PLMs_cmds/run_blastp.sh -q ../data/db/astral_originrosetta.fa  -t ../data/db/astral.fa  -k 15177 -p 15177
# ... repeat for diamond, mmseq2, near, plm, tmvec, dctdomain
bash PLMs_cmds/run_dhr.sh    -q ../data/db/astral_originrosetta.tsv -t ../data/db/astral.tsv -k 15177 -p 15177
```

**Step 4 — Run sanity check:**

```bash
python sanity_check/check_bio_mut_structure.py
```

Output saved to `results/check_bio_structure/`.

---

### Experiment 3 — Redundancy Stability

Probes whether similarity scores degrade gracefully when query sequences are artificially doubled (self-concatenated or shuffle-concatenated).

**Run search methods:**

```bash
source PLMs_cmds/.env

# Run for each query in {astral.fa, astral_doubshuf.fa, astral_doubself.fa}; target: astral.fa
# -k/-p: see table above (dctdomain: -k 3000 -p 1000; all others: -k 1000 -p 1000)
bash PLMs_cmds/run_blastp.sh -q ../data/db/astral.fa  -t ../data/db/astral.fa  -k 1000 -p 1000
# ... repeat for diamond, mmseq2, near, plm, tmvec, dctdomain
bash PLMs_cmds/run_dhr.sh    -q ../data/db/astral.tsv -t ../data/db/astral.tsv -k 1000 -p 1000
```

**Run sanity check:**

```bash
python sanity_check/check_pert_doublen.py
```

Output saved to `results/check_pert_doublen/`.

---

### Experiment 4 — Similarity Monotonicity

Tests whether similarity scores decrease monotonically as query sequences are progressively truncated.

**Run search methods:**

```bash
source PLMs_cmds/.env

# Run for each query in {astral.fa, astral_trunchalf.fa, astral_truncqrt.fa}; target: astral.fa
# -k/-p: see table above (dctdomain: -k 3000 -p 1000; all others: -k 1000 -p 1000)
bash PLMs_cmds/run_blastp.sh -q ../data/db/astral.fa  -t ../data/db/astral.fa  -k 1000 -p 1000
# ... repeat for diamond, mmseq2, near, plm, tmvec, dctdomain
bash PLMs_cmds/run_dhr.sh    -q ../data/db/astral.tsv -t ../data/db/astral.tsv -k 1000 -p 1000
```

**Run sanity check:**

```bash
python sanity_check/check_pert_truncation.py
```

Output saved to `results/check_pert_truncation/`.

---

### Experiment 5 — Decoy Sensitivity

Evaluates whether methods can distinguish true homologs from permuted-data decoys (sequences with destroyed biological signal).

**Run search against a combined database of standard and shuffled sequences:**

```bash
source PLMs_cmds/.env

# Query: astral.fa | Target: astral_shuf_ori.fa
# -k/-p: see table above (dctdomain: -k 91062 -p 30354; all others: -k 30354 -p 30354)
bash PLMs_cmds/run_blastp.sh -q ../data/db/astral.fa  -t ../data/db/astral_shuf_ori.fa  -k 30354 -p 30354
# ... repeat for near, plm, tmvec, dctdomain
bash PLMs_cmds/run_dhr.sh    -q ../data/db/astral.tsv -t ../data/db/astral_shuf_ori.tsv -k 30354 -p 30354
```

**Run sanity check:**

```bash
python sanity_check/check_perm_data.py
```

Output saved to `results/check_perm_data/`.

---

### Experiment 6 — Representation Reliability

Tests whether PLM-based methods produce consistent representations by comparing standard embeddings against permuted-model variants.

**Run standard search against database (baseline):**

We directly reuse the search results obtained from Experiment 5.

**Run permuted-model search:**

```bash
# Query: astral.fa | Target: astral.fa
# -k/-p: see table above (dctdomain: -k 45531 -p 15177; all others: -k 15177 -p 15177)
bash PLMs_cmds/run_perm_plm.sh -q ../data/db/astral.fa  -t ../data/db/astral.fa  -k 15177 -p 15177
# ... repeat for tmvec, dctdomain
bash PLMs_cmds/run_perm_dhr.sh -q ../data/db/astral.tsv -t ../data/db/astral.tsv -k 15177 -p 15177
```

**Run sanity check:**

```bash
python sanity_check/check_perm_model.py
```

Output saved to `results/check_perm_model/`.

---

### Overall Performance Summary

After running all six experiments, generate the combined bubble plot:

```bash
python sanity_check/overall_performance.py
```

Output saved to `results/overall_performance/overall_performance_bubble.pdf`.
