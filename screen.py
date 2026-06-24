#!/usr/bin/env python3
"""Screen a relaxed .extxyz file and write the top-N viable structures.

Viability filters (all configurable via CLI):
  --vol-min / --vol-max   volume per atom in Å³  (default 5–60)
  --min-dist              minimum pairwise interatomic distance (default 1.5 Å)

Ranking (--sort-by):
  min_dist     highest min-distance first — least clashing structures
  vol_per_atom lowest volume-per-atom first — densest viable structures
  none         preserve original file order

Usage
-----
    python screen.py relaxed.extxyz screened.extxyz -n 5000
    python screen.py relaxed.extxyz screened.extxyz -n 5000 --vol-max 50 --min-dist 1.8
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Iterator

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------------
# extxyz frame
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    n_atoms: int
    comment: str
    species: list[str]
    positions: np.ndarray  # (n_atoms, 3) Cartesian Å
    lattice: np.ndarray    # (3, 3), rows = lattice vectors

    @property
    def volume(self) -> float:
        return abs(float(np.linalg.det(self.lattice)))

    @property
    def vol_per_atom(self) -> float:
        return self.volume / self.n_atoms

    def min_dist(self) -> float:
        """Minimum pairwise distance with PBC via minimum-image convention."""
        if self.n_atoms == 1:
            return float("inf")
        L_inv = np.linalg.inv(self.lattice)
        frac = self.positions @ L_inv              # (n, 3) fractional
        df = frac[:, None, :] - frac[None, :, :]  # (n, n, 3) displacements
        df -= np.round(df)                         # minimum image
        cart = df @ self.lattice                   # back to Cartesian
        d2 = np.sum(cart ** 2, axis=-1)            # (n, n)
        np.fill_diagonal(d2, np.inf)
        return float(d2.min() ** 0.5)

    def to_extxyz_lines(self) -> str:
        lines = [str(self.n_atoms), self.comment]
        for sp, (x, y, z) in zip(self.species, self.positions):
            lines.append(f"{sp}  {x:.8f}  {y:.8f}  {z:.8f}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_LATTICE_RE = re.compile(r'Lattice="([^"]+)"')


def _parse_lattice(comment: str) -> np.ndarray:
    m = _LATTICE_RE.search(comment)
    if not m:
        raise ValueError(f"No Lattice= key found in comment: {comment[:80]!r}")
    return np.array(list(map(float, m.group(1).split()))).reshape(3, 3)


def iter_frames(path: str) -> Iterator[Frame]:
    """Stream frames one at a time from an extxyz file.

    Silently skips truncated or malformed frames (e.g. last frame if the
    relaxation run was interrupted before the file was fully written).
    """
    with open(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            try:
                n = int(header.strip())
            except ValueError:
                break
            comment = fh.readline().rstrip("\n")
            try:
                lattice = _parse_lattice(comment)
            except ValueError:
                # Skip n atom lines and move on
                for _ in range(n):
                    fh.readline()
                continue

            species: list[str] = []
            positions: list[list[float]] = []
            ok = True
            for _ in range(n):
                parts = fh.readline().split()
                if len(parts) < 4:
                    ok = False
                    break
                try:
                    species.append(parts[0])
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    ok = False
                    break
            if not ok:
                continue
            yield Frame(
                n_atoms=n,
                comment=comment,
                species=species,
                positions=np.array(positions, dtype=np.float64),
                lattice=lattice,
            )


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

def screen(
    input_path: str,
    output_path: str,
    top_n: int,
    vol_min: float,
    vol_max: float,
    min_dist_threshold: float,
    sort_by: str,
) -> None:
    candidates: list[tuple[float, float, Frame]] = []
    total = 0
    skipped_vol = 0
    skipped_dist = 0

    raw_stream = iter_frames(input_path)
    stream = tqdm(raw_stream, desc="Screening", unit="structures") if tqdm is not None else raw_stream

    for frame in stream:
        total += 1
        vpa = frame.vol_per_atom
        if not (vol_min <= vpa <= vol_max):
            skipped_vol += 1
            continue
        md = frame.min_dist()
        if md < min_dist_threshold:
            skipped_dist += 1
            continue
        candidates.append((vpa, md, frame))

    print(
        f"\nRead {total} structures: "
        f"{skipped_vol} failed vol filter, "
        f"{skipped_dist} failed min-dist filter, "
        f"{len(candidates)} passed.",
        file=sys.stderr,
    )

    if sort_by == "min_dist":
        candidates.sort(key=lambda t: -t[1])
    elif sort_by == "vol_per_atom":
        candidates.sort(key=lambda t: t[0])
    # "none" → preserve original order

    selected = candidates[:top_n]
    print(f"Writing top {len(selected)} to {output_path!r}", file=sys.stderr)

    with open(output_path, "w") as out:
        for _, _, frame in selected:
            out.write(frame.to_extxyz_lines())

    if len(candidates) < top_n:
        print(
            f"Note: only {len(candidates)} structures passed all filters "
            f"(requested {top_n}).",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input",  help="Input .extxyz file (relaxed structures)")
    p.add_argument("output", help="Output .extxyz file (screened structures)")
    p.add_argument(
        "-n", "--top-n", type=int, default=1000, metavar="N",
        help="Maximum number of structures to write (default: 1000)",
    )
    p.add_argument(
        "--vol-min", type=float, default=5.0, metavar="Å³",
        help="Minimum volume per atom in Å³ (default: 5.0)",
    )
    p.add_argument(
        "--vol-max", type=float, default=60.0, metavar="Å³",
        help="Maximum volume per atom in Å³ (default: 60.0)",
    )
    p.add_argument(
        "--min-dist", type=float, default=1.5, metavar="Å",
        help="Minimum allowed interatomic distance in Å (default: 1.5)",
    )
    p.add_argument(
        "--sort-by",
        choices=["min_dist", "vol_per_atom", "none"],
        default="min_dist",
        help=(
            "Ranking criterion for top-N selection: "
            "'min_dist' (default, highest first), "
            "'vol_per_atom' (lowest first), "
            "'none' (original order)"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    screen(
        input_path=args.input,
        output_path=args.output,
        top_n=args.top_n,
        vol_min=args.vol_min,
        vol_max=args.vol_max,
        min_dist_threshold=args.min_dist,
        sort_by=args.sort_by,
    )


if __name__ == "__main__":
    main()
