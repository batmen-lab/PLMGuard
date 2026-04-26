#!/usr/bin/env python3
"""Analyze mut60p relaxed models with PyRosetta-based structural sanity metrics.

Metrics:
- Total energy with ref2015 full-atom score function
- fa_rep and fa_rep per residue (clash indicator)
- rama_prepro and omega (backbone torsion sanity)
- H-bond terms (sr/lr/bb_sc/sc + totals)
- CA RMSD to the original structure

Both mutated and original PDBs live under the same data dir:
  data/rosetta_mut/<id>/<id>_mut60p_relax.pdb   (mutated)
  data/rosetta_mut/<id>/<id>.pdb                 (original)

Quality criterion: CA RMSD < 3.0 → acceptable; >= 3.0 → rejected.
Other metrics (fa_rep, torsion, hbond) are recorded as notes only.

Outputs (named from --fa-file):
  <stem>.quality.csv       full per-model scoring table
  <stem>_filtered.fa       FASTA with rejected sequences removed
"""

from __future__ import annotations

import argparse, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def extract_sequence_id(mut_pdb: Path) -> str:
    # d2vz8a3_mut60p_relax.pdb -> d2vz8a3_mut60p  (matches FASTA header)
    name = mut_pdb.name
    if name.endswith("_relax.pdb"):
        return name[: -len("_relax.pdb")]
    return mut_pdb.stem


def get_reference_path(mut_pdb: Path) -> Path:
    # Original PDB sits in the same folder, named after the folder itself.
    return mut_pdb.parent / f"{mut_pdb.parent.name}.pdb"


def classify_row(
    fa_rep_per_res: float,
    rama_prepro: float,
    omega: float,
    backbone_hbond_total: float,
    hbond_sc: float,
    ca_rmsd: float | None,
) -> tuple[str, str]:
    notes: list[str] = []

    if fa_rep_per_res >= 3.0:
        notes.append("fa_rep_per_res>=3.0")
    elif fa_rep_per_res >= 1.5:
        notes.append("fa_rep_per_res>=1.5")

    if rama_prepro >= 10.0:
        notes.append("rama_prepro>=10")
    if omega >= 5.0:
        notes.append("omega>=5")

    if backbone_hbond_total >= 0.0:
        notes.append("backbone_hbond>=0")
    if hbond_sc > 10.0:
        notes.append("hbond_sc_high")

    rejected = ca_rmsd is not None and ca_rmsd >= 3.0
    if rejected:
        notes.append(f"CA_RMSD={ca_rmsd:.2f}>=3")

    overall = "rejected" if rejected else "acceptable"
    if not notes:
        notes.append("pass")
    return overall, ";".join(notes)


def plot_ca_rmsd_distribution(df: pd.DataFrame, out_path: Path) -> None:
    values = df["ca_rmsd_to_ref"].dropna().values
    if len(values) == 0:
        return

    bins = np.linspace(0, 10, 31)
    counts, edges = np.histogram(values, bins=bins)

    fig, ax = plt.subplots(figsize=(5, 5))
    added_kept = added_removed = False
    for c, left, right in zip(counts, edges[:-1], edges[1:]):
        if c == 0:
            continue
        is_kept = right <= 3
        color = "#6baed6" if is_kept else "0.6"
        label = None
        if is_kept and not added_kept:
            label = "Similar (<3 Å, kept)";  added_kept = True
        if not is_kept and not added_removed:
            label = "Different (≥3 Å, removed)";  added_removed = True
        ax.bar(left, c, width=right - left, align="edge", color=color, edgecolor="white", alpha=0.85, label=label)

    ax.axvline(3, color="black", linestyle="--", linewidth=1.0, label="Threshold (3 Å)")
    ax.set_xlim(0, 10)
    ax.set_xlabel("Ca-RMSD to Original (Å)")
    ax.set_ylabel("Count")
    ax.set_title("Ca-RMSD Distribution")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_farep_kde(df: pd.DataFrame, out_path: Path) -> None:
    kept = df[df["ca_rmsd_to_ref"].notna() & (df["ca_rmsd_to_ref"] < 3.0)]
    original = kept["fa_rep_per_res_original"].dropna().values
    mutated  = kept["fa_rep_per_res_mut"].dropna().values
    if len(original) == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.kdeplot(original, ax=ax, color="#6baed6", label="Original Structure", fill=True)
    sns.kdeplot(mutated,  ax=ax, color="#e7ba52", label="Mutated Structure",  fill=True)
    ax.set_xlim(0, 10)
    ax.set_xlabel("fa_rep Per Residue")
    ax.set_ylabel("Density")
    ax.set_title("fa_rep Per Residue (Ca-RMSD < 3 Å)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def analyze_models(
    data_dir: Path, fa_file: Path, astral_fa: Path, mute_rosetta: bool = True
) -> None:
    out_csv = fa_file.with_suffix(".quality.csv")

    if out_csv.exists():
        print(f"CSV found, skipping scoring: {out_csv}")
        df = pd.read_csv(out_csv)
        failed = []
    else:
        df, failed = _score_models(data_dir, mute_rosetta)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"CSV      : {out_csv}")

    accepted_ids = set(df.loc[df["overall_quality"] != "rejected", "sequence_id"])
    out_fa = fa_file.with_name("astral_mutrosetta.fa")
    out_fa.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with fa_file.open() as fin, out_fa.open("w") as fout:
        write = False
        for line in fin:
            if line.startswith(">"):
                seq_id = line[1:].split()[0]
                write = seq_id in accepted_ids
                if write:
                    kept += 1
            if write:
                fout.write(line)
    print(f"Filtered : {out_fa}  ({kept} sequences written)")

    origin_ids = {sid.replace("_mut60p", "") for sid in accepted_ids}
    out_origin_fa = fa_file.with_name("astral_originrosetta.fa")
    kept_origin = 0
    with astral_fa.open() as fin, out_origin_fa.open("w") as fout:
        write = False
        for line in fin:
            if line.startswith(">"):
                seq_id = line[1:].split()[0]
                write = seq_id in origin_ids
                if write:
                    kept_origin += 1
            if write:
                fout.write(line)
    print(f"Origin   : {out_origin_fa}  ({kept_origin} sequences written)")

    if failed:
        print(f"Failed   : {len(failed)} PDBs — check stderr or add --verbose-rosetta")
    return df

def _score_models(
    data_dir: Path, mute_rosetta: bool
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    try:
        import pyrosetta
        from pyrosetta import get_fa_scorefxn, init, pose_from_pdb
        from pyrosetta.rosetta.core.scoring import CA_rmsd, ScoreType, bb_rmsd
    except Exception as exc:
        msg = str(exc)
        if "numpy.dtype size changed" in msg or "binary incompatibility" in msg:
            raise SystemExit(
                "PyRosetta import failed due to NumPy/h5py binary incompatibility.\n"
                "In your conda env, reinstall compatible numpy+h5py, e.g.:\n"
                "  conda install -n ADTnormPy -c conda-forge \"numpy=1.26.*\" \"h5py>=3.10,<3.12\" --force-reinstall\n"
                "Then retry this script."
            ) from exc
        raise SystemExit(
            "PyRosetta is required. Please run in an environment with pyrosetta installed."
        ) from exc

    base_opts = "-ignore_unrecognized_res 1 -load_PDB_components false"
    opts = f"-mute all {base_opts}" if mute_rosetta else base_opts
    try:
        already_init = pyrosetta.rosetta.basic.was_init_called()
    except Exception:
        already_init = False
    if not already_init:
        try:
            init(opts)
        except TypeError:
            init(extra_options=opts)
    scorefxn = get_fa_scorefxn()

    def pose_metrics(pose) -> dict[str, float]:
        total_energy = float(scorefxn(pose))
        emap = pose.energies().total_energies()

        def term(name: str) -> float:
            try:
                st = getattr(ScoreType, name)
            except AttributeError as exc:
                raise KeyError(f"Unknown Rosetta score term: {name}") from exc
            return float(emap[st])

        num_res = int(pose.size())
        fa_rep = term("fa_rep")
        hbond_sr_bb = term("hbond_sr_bb")
        hbond_lr_bb = term("hbond_lr_bb")
        hbond_bb_sc = term("hbond_bb_sc")
        hbond_sc = term("hbond_sc")
        backbone_hbond_total = hbond_sr_bb + hbond_lr_bb + hbond_bb_sc
        total_hbond = backbone_hbond_total + hbond_sc

        return {
            "num_residues": num_res,
            "total_energy": total_energy,
            "total_energy_per_res": (total_energy / num_res) if num_res > 0 else float("nan"),
            "fa_rep": fa_rep,
            "fa_rep_per_res": (fa_rep / num_res) if num_res > 0 else float("nan"),
            "rama_prepro": term("rama_prepro"),
            "omega": term("omega"),
            "hbond_sr_bb": hbond_sr_bb,
            "hbond_lr_bb": hbond_lr_bb,
            "hbond_bb_sc": hbond_bb_sc,
            "hbond_sc": hbond_sc,
            "backbone_hbond_total": backbone_hbond_total,
            "total_hbond": total_hbond,
        }

    target_pdbs = sorted(data_dir.rglob("*_mut60p_relax.pdb"))
    if not target_pdbs:
        raise SystemExit(f"No *_mut60p_relax.pdb found under {data_dir}")

    records: list[dict] = []
    failed: list[tuple[str, str]] = []

    for pdb_path in target_pdbs:
        try:
            seq_id = extract_sequence_id(pdb_path)
            pose_mut = pose_from_pdb(str(pdb_path))
            mut_metrics = pose_metrics(pose_mut)

            ref_path = get_reference_path(pdb_path)
            ca_rmsd = None
            backbone_rmsd = None
            rmsd_note = "ref_missing"
            ref_metrics: dict[str, float] | None = None
            if ref_path.exists():
                pose_ref = pose_from_pdb(str(ref_path))
                if int(pose_ref.size()) == int(mut_metrics["num_residues"]):
                    ref_metrics = pose_metrics(pose_ref)
                    ca_rmsd = float(CA_rmsd(pose_ref, pose_mut))
                    backbone_rmsd = float(bb_rmsd(pose_ref, pose_mut))
                    rmsd_note = "ok"
                else:
                    rmsd_note = f"size_mismatch:{pose_ref.size()}vs{int(mut_metrics['num_residues'])}"

            overall, notes = classify_row(
                fa_rep_per_res=float(mut_metrics["fa_rep_per_res"]),
                rama_prepro=float(mut_metrics["rama_prepro"]),
                omega=float(mut_metrics["omega"]),
                backbone_hbond_total=float(mut_metrics["backbone_hbond_total"]),
                hbond_sc=float(mut_metrics["hbond_sc"]),
                ca_rmsd=ca_rmsd,
            )

            records.append({
                "sequence_id": seq_id,
                "mut60p_relax_pdb": str(pdb_path),
                "reference_pdb": str(ref_path) if ref_path.exists() else None,
                "num_residues": int(mut_metrics["num_residues"]),
                "total_energy_mut": mut_metrics["total_energy"],
                "total_energy_original": ref_metrics["total_energy"] if ref_metrics else None,
                "total_energy_per_res_mut": mut_metrics["total_energy_per_res"],
                "total_energy_per_res_original": ref_metrics["total_energy_per_res"] if ref_metrics else None,
                "fa_rep_mut": mut_metrics["fa_rep"],
                "fa_rep_original": ref_metrics["fa_rep"] if ref_metrics else None,
                "fa_rep_per_res_mut": mut_metrics["fa_rep_per_res"],
                "fa_rep_per_res_original": ref_metrics["fa_rep_per_res"] if ref_metrics else None,
                "rama_prepro_mut": mut_metrics["rama_prepro"],
                "rama_prepro_original": ref_metrics["rama_prepro"] if ref_metrics else None,
                "omega_mut": mut_metrics["omega"],
                "omega_original": ref_metrics["omega"] if ref_metrics else None,
                "hbond_sr_bb_mut": mut_metrics["hbond_sr_bb"],
                "hbond_sr_bb_original": ref_metrics["hbond_sr_bb"] if ref_metrics else None,
                "hbond_lr_bb_mut": mut_metrics["hbond_lr_bb"],
                "hbond_lr_bb_original": ref_metrics["hbond_lr_bb"] if ref_metrics else None,
                "hbond_bb_sc_mut": mut_metrics["hbond_bb_sc"],
                "hbond_bb_sc_original": ref_metrics["hbond_bb_sc"] if ref_metrics else None,
                "hbond_sc_mut": mut_metrics["hbond_sc"],
                "hbond_sc_original": ref_metrics["hbond_sc"] if ref_metrics else None,
                "backbone_hbond_total_mut": mut_metrics["backbone_hbond_total"],
                "backbone_hbond_total_original": ref_metrics["backbone_hbond_total"] if ref_metrics else None,
                "total_hbond_mut": mut_metrics["total_hbond"],
                "total_hbond_original": ref_metrics["total_hbond"] if ref_metrics else None,
                "ca_rmsd_to_ref": ca_rmsd,
                "backbone_rmsd_to_ref": backbone_rmsd,
                "rmsd_note": rmsd_note,
                "overall_quality": overall,
                "notes": notes,
            })
        except Exception as exc:
            failed.append((str(pdb_path), str(exc)))

    return pd.DataFrame(records), failed


def plot_tmscore(df: pd.DataFrame, data_dir: Path, out_path: Path) -> None:
    # Load TM-scores from mut_summary.csv (mut_percent=60) for each model folder.
    tm_by_folder: dict[str, float] = {}
    for csv_path in data_dir.rglob("mut_summary.csv"):
        try:
            summary = pd.read_csv(csv_path)
            row = summary[summary["mut_percent"] == 60]
            if row.empty:
                continue
            tm_by_folder[csv_path.parent.name] = float(row["TM_score (origin vs mutrelax)"].iloc[0])
        except Exception:
            continue

    # Restrict to kept models (ca_rmsd_to_ref < 3.0); folder name = sequence_id "_mut60p".
    kept = df[df["ca_rmsd_to_ref"].notna() & (df["ca_rmsd_to_ref"] < 3.0)]
    folders = kept["sequence_id"].str.replace("_mut60p", "", regex=False)
    tm_scores = [tm_by_folder[f] for f in folders if f in tm_by_folder]
    if not tm_scores:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.histplot(tm_scores, bins=30, ax=ax, color="#6baed6", alpha=0.8, edgecolor="white")
    ax.set_xlabel("TM-Score")
    ax.set_ylabel("Count")
    ax.set_title("TM-Score: Original vs Mutated (Ca-RMSD < 3 Å)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data/rosetta_mut"),
        help="Directory containing mutated (*_mut60p_relax.pdb) and original (<id>.pdb) structures",
    )
    parser.add_argument(
        "--fa-file",
        type=Path,
        default=Path("../data/rosetta_mut/astral_s1000_mutrosetta.fa"),
        help="Raw FASTA file to filter",
    )
    parser.add_argument(
        "--verbose-rosetta",
        action="store_true",
        help="Do not mute Rosetta logs",
    )
    args = parser.parse_args()
    
    pparent_dir = Path(__file__).resolve().parent.parent.parent
    
    # generate the CSV and filtered FASTA
    astral_fa = pparent_dir / "data/db/astral.fa"
    df = analyze_models(args.data_dir, args.fa_file, astral_fa, mute_rosetta=not args.verbose_rosetta)
    
    # generate plots
    out_dir = pparent_dir / "results/check_bio_structure"
    os.makedirs(out_dir, exist_ok=True)
    
    plot_ca_rmsd_distribution(df, out_dir / "ca_rmsd_distribution.pdf")
    plot_farep_kde(df, out_dir / "farep_kde.pdf")
    plot_tmscore(df, args.data_dir, out_dir / "tmcore.pdf")
    
if __name__ == "__main__":
    main()

# python utils/filter_mut_rosetta.py
