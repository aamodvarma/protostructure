"""
Sample 10 random structures from an extxyz file and write each to a CIF file.

Usage:
    python sample_extxyz_to_cif.py <input.extxyz> [--out-dir DIR] [--n N] [--seed S]
"""

import argparse
import random
from pathlib import Path

from ase.io import read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to the extxyz file")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("sampled_cifs"),
        help="Output directory for CIF files",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of structures to sample"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="RNG seed for reproducibility"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: count frames cheaply without holding everything in memory.
    # ase.io.iread streams frames one at a time.
    from ase.io import iread

    n_total = sum(1 for _ in iread(args.input, index=":", format="extxyz"))
    if n_total == 0:
        raise SystemExit(f"No structures found in {args.input}")
    if args.n > n_total:
        raise SystemExit(f"Requested {args.n} but file only has {n_total} structures")

    # Pick indices first, then read only those frames.
    picked = sorted(random.sample(range(n_total), args.n))
    print(f"Sampling indices {picked} from {n_total} total structures")

    # Read just the selected frames. Using a single call with a list of indices
    # would require ase >= 3.23; iterating with iread is portable.
    selected = []
    picked_set = set(picked)
    for i, atoms in enumerate(iread(args.input, index=":", format="extxyz")):
        if i in picked_set:
            selected.append((i, atoms))
        if len(selected) == len(picked):
            break

    for orig_idx, atoms in selected:
        out_path = args.out_dir / f"structure_{orig_idx:06d}.cif"
        write(out_path, atoms, format="cif")
        print(
            f"  wrote {out_path} ({atoms.get_chemical_formula()}, {len(atoms)} atoms)"
        )

    print(f"Done. {len(selected)} CIFs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
