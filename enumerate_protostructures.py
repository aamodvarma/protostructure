#!/usr/bin/env python3
"""Enumerate random ternary protostructure labels following the Aviary/AFLOW convention.

Protostructure label format (Aviary convention):
    {anon_formula}_{pearson_symbol}_{spg_num}_{wyckoff_letters}:{el1}-{el2}-{el3}

Example:
    ABC_cF12_225_a_b_c:Cl-K-Na

Reference:
    "Screening 39 billion protostructures for materials discovery"
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import random
import periodictable
import sys
import time
from collections import Counter
from functools import reduce
from itertools import combinations_with_replacement
from math import gcd
from string import ascii_uppercase

from pymatgen.analysis.prototypes import (
    CRYSTAL_FAMILY_SYMBOLS,
    RE_SUBST_ONE_PREFIX,
    RE_WYCKOFF,
    WYCKOFF_MULTIPLICITY_DICT,
    canonicalize_element_wyckoffs,
)
from pymatgen.symmetry.groups import SpaceGroup

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Element pool: Z < 90, noble gases excluded
# ---------------------------------------------------------------------------

# what mace is trained on
l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91, 92, 93, 94]
noble_gas = [2, 10, 18, 36, 54, 86, 118]

_ELEMENTS_POOL: list[str] = [periodictable.elements[i].symbol for i in l if i < 90 and i not in noble_gas] 

# ---------------------------------------------------------------------------
# Pre-computed space-group metadata
# ---------------------------------------------------------------------------

# Pearson symbol: base-centered A/B/C → S
_BRAVAIS_REMAP: dict[str, str] = {"A": "S", "B": "S", "C": "S"}


def _pearson_prefix(sg_num: int) -> str:
    sg = SpaceGroup.from_int_number(sg_num)
    family = CRYSTAL_FAMILY_SYMBOLS[sg.crystal_system]
    bravais = _BRAVAIS_REMAP.get(sg.symbol[0], sg.symbol[0])
    return family + bravais


def _build_sg_table() -> dict[int, dict]:
    """Pre-compute per-space-group metadata (letters, multiplicities, Pearson prefix)."""
    table: dict[int, dict] = {}
    for sg in range(1, 231):
        sg_str = str(sg)
        if sg_str not in WYCKOFF_MULTIPLICITY_DICT:
            continue
        wycks = WYCKOFF_MULTIPLICITY_DICT[sg_str]
        if not wycks:
            continue
        letters = sorted(wycks)
        table[sg] = {
            "letters": letters,
            "mult": {l: wycks[l] for l in letters},
            "pearson_prefix": _pearson_prefix(sg),
        }
    return table


# Build once at import time
_SG_TABLE: dict[int, dict] = _build_sg_table()
_SG_LIST: list[int] = list(_SG_TABLE)


# ---------------------------------------------------------------------------
# Label-building helpers
# ---------------------------------------------------------------------------

def _wyckoff_str(letters: list[str] | tuple[str, ...]) -> str:
    """Build AFLOW-format Wyckoff string from a (possibly repeated) list of letters.

    Examples:
        ['a', 'a', 'b'] → '2ab'
        ['d', 'g']      → 'dg'
        ['c']           → 'c'
    """
    counts = Counter(letters)
    result = ""
    for letter in sorted(counts):
        cnt = counts[letter]
        result += (str(cnt) if cnt > 1 else "") + letter
    return result


def _preprocess_ew(element_wyckoffs: str) -> str:
    """Add '1' prefix to bare Wyckoff letters so canonicalize_element_wyckoffs works.

    Converts 'dg' → '1d1g', '2ab' → '2a1b', etc.
    Applied per-element (underscore-separated) section.
    """
    return "_".join(
        RE_WYCKOFF.sub(RE_SUBST_ONE_PREFIX, part)
        for part in element_wyckoffs.split("_")
    )


def _anon_formula(reduced_counts: list[int]) -> str:
    """Build anonymous formula string from GCD-reduced element counts.

    Elements are assigned letters A, B, C in alphabetical order of element
    symbol (which matches the sorted-elements ordering used in this module).

    Examples:
        [1, 1, 1] → 'ABC'
        [1, 2, 1] → 'AB2C'
        [2, 3, 1] → 'A2B3C'
    """
    formula = ""
    for letter, amt in zip(ascii_uppercase, reduced_counts):
        formula += letter + ("" if amt == 1 else str(amt))
    return formula


def _make_label(
    sg: int,
    wyck_slots: list[tuple[str, ...]],
    sorted_elements: list[str],
    sg_data: dict,
) -> str | None:
    """Build a canonicalized protostructure label for any number of elements.

    Args:
        sg: Space group number.
        wyck_slots: One sorted tuple of Wyckoff letters per element, in the
            same order as sorted_elements.
        sorted_elements: Alphabetically sorted list of element symbols.
        sg_data: Pre-computed metadata dict for this space group.

    Returns:
        Protostructure label string, or None if canonicalization fails.
    """
    mult = sg_data["mult"]
    prefix = sg_data["pearson_prefix"]

    ew_raw = "_".join(_wyckoff_str(slot) for slot in wyck_slots)
    ew_pre = _preprocess_ew(ew_raw)

    try:
        canon_ew = canonicalize_element_wyckoffs(ew_pre, sg)
    except Exception:
        return None

    counts = [sum(mult[w] for w in slot) for slot in wyck_slots]
    total_atoms = sum(counts)
    g = reduce(gcd, counts)
    anon = _anon_formula([c // g for c in counts])

    pearson = prefix + str(total_atoms)
    chem_sys = "-".join(sorted_elements)
    return f"{anon}_{pearson}_{sg}_{canon_ew}:{chem_sys}"


# ---------------------------------------------------------------------------
# Partition helpers
# ---------------------------------------------------------------------------

def _partitions_n(total: int, n: int) -> list[tuple[int, ...]]:
    """All ordered n-tuples of positive integers summing to total."""
    if n == 1:
        return [(total,)] if total >= 1 else []
    result = []
    for first in range(1, total - n + 2):
        for rest in _partitions_n(total - first, n - 1):
            result.append((first,) + rest)
    return result


# ---------------------------------------------------------------------------
# Random sampling (primary mode)
# ---------------------------------------------------------------------------

def random_sample(
    elements: list[str] | None,
    n_samples: int,
    max_complexity: int = 5,
    seed: int | None = None,
    max_attempts_multiplier: int = 200,
) -> list[str]:
    """Return a random subset of protostructure labels.

    Uses rejection sampling: generates random (sg, complexity, wyckoff) tuples
    until ``n_samples`` unique canonical labels are collected.

    Args:
        elements: Element symbols to use for every sample.  Pass ``None`` to
            enable *random-elements mode*: each iteration independently draws a
            count n ∈ {2, 3, 4, 5} uniformly at random, then samples n distinct
            elements from the periodic-table pool (Z < 90, no noble gases).
        n_samples: Number of unique protostructure labels to return.
        max_complexity: Maximum total Wyckoff sites across all elements
            (minimum equals the number of elements, one site each).
        seed: Random seed for reproducibility.
        max_attempts_multiplier: Stop after this multiple of n_samples attempts.

    Returns:
        Sorted list of unique protostructure label strings.
    """
    fixed_elements: list[str] | None = None
    if elements is not None:
        _validate_elements(elements)
        fixed_elements = sorted(elements)

    rng = random.Random(seed)
    labels: set[str] = set()
    max_attempts = max_attempts_multiplier * n_samples

    attempts = 0
    while len(labels) < n_samples and attempts < max_attempts:
        attempts += 1

        # --- Choose elements for this iteration ---
        if fixed_elements is not None:
            sorted_els = fixed_elements
        else:
            n_els = rng.randint(2, 5)
            sorted_els = sorted(rng.sample(_ELEMENTS_POOL, n_els))

        n_els = len(sorted_els)
        min_complexity = n_els  # at least one Wyckoff site per element

        if max_complexity < min_complexity:
            continue

        sg = rng.choice(_SG_LIST)
        d = _SG_TABLE[sg]
        letters = d["letters"]

        total_sites = rng.randint(min_complexity, max_complexity)
        partition = rng.choice(_partitions_n(total_sites, n_els))

        wyck_slots = [
            tuple(sorted(rng.choices(letters, k=k))) for k in partition
        ]

        label = _make_label(sg, wyck_slots, sorted_els, d)
        if label is not None:
            labels.add(label)

    if len(labels) < n_samples:
        log.warning(
            "Only %d unique labels found after %d attempts (requested %d). "
            "Try lowering n_samples or raising max_attempts_multiplier.",
            len(labels),
            attempts,
            n_samples,
        )

    return sorted(labels)


# ---------------------------------------------------------------------------
# Full enumeration (exhaustive)
# ---------------------------------------------------------------------------

def _enumerate_sg(args: tuple) -> list[str]:
    """Worker function: enumerate all labels for one space group."""
    sg, sorted_els, max_complexity = args
    if sg not in _SG_TABLE:
        return []

    d = _SG_TABLE[sg]
    letters = d["letters"]
    n_els = len(sorted_els)
    results: set[str] = set()

    from itertools import product as iproduct
    for total_sites in range(n_els, max_complexity + 1):
        for partition in _partitions_n(total_sites, n_els):
            for wyck_slots in iproduct(*[
                combinations_with_replacement(letters, k) for k in partition
            ]):
                label = _make_label(sg, list(wyck_slots), sorted_els, d)
                if label is not None:
                    results.add(label)

    return list(results)


def enumerate_all(
    elements: list[str],
    max_complexity: int = 5,
    n_workers: int | None = None,
    show_progress: bool = True,
) -> list[str]:
    """Exhaustively enumerate all ternary protostructure labels.

    This is the full enumeration (~83M raw combinations across all 230 space
    groups for complexity 5).  Parallelized over space groups.

    Args:
        elements: Exactly 3 distinct element symbols.
        max_complexity: Maximum Wyckoff sites across all 3 elements.
        n_workers: Number of parallel workers (default: all CPU cores).
        show_progress: Print progress to stderr.

    Returns:
        Sorted list of unique protostructure label strings.
    """
    _validate_elements(elements)
    sorted_els = sorted(elements)

    if n_workers is None:
        n_workers = mp.cpu_count()

    args = [(sg, sorted_els, max_complexity) for sg in _SG_LIST]

    if show_progress:
        print(
            f"Full enumeration: {len(_SG_LIST)} space groups, "
            f"max complexity {max_complexity}, {n_workers} workers",
            file=sys.stderr,
        )

    t0 = time.time()
    all_labels: set[str] = set()

    with mp.Pool(n_workers) as pool:
        for i, partial in enumerate(pool.imap_unordered(_enumerate_sg, args), 1):
            all_labels.update(partial)
            if show_progress and i % 20 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                remaining = (len(_SG_LIST) - i) / rate
                print(
                    f"  [{i}/{len(_SG_LIST)}] {len(all_labels):,} labels "
                    f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)",
                    file=sys.stderr,
                )

    if show_progress:
        print(
            f"Done: {len(all_labels):,} unique labels in {time.time()-t0:.1f}s",
            file=sys.stderr,
        )

    return sorted(all_labels)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_elements(elements: list[str]) -> None:
    if not (2 <= len(elements) <= 5):
        raise ValueError(f"Between 2 and 5 elements required, got {len(elements)}.")
    if len(set(elements)) != len(elements):
        raise ValueError(f"All elements must be distinct, got {elements}.")
    for el in elements:
        if not el.isalpha():
            raise ValueError(f"Invalid element symbol: {el!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "elements",
        nargs="*",
        metavar="ELEMENT",
        help="2–5 element symbols (e.g. Na Cl K).  Omit when --random-elements is used.",
    )
    p.add_argument(
        "--random-elements",
        action="store_true",
        help="For each random sample, draw a count n ∈ {2,3,4,5} uniformly at random "
             "then pick n distinct elements from the periodic table (Z < 90, no noble "
             "gases).  Incompatible with positional ELEMENT arguments.",
    )
    p.add_argument(
        "-n", "--n-samples",
        type=int,
        default=1000,
        metavar="N",
        help="Number of random protostructure labels to generate (default: 1000). "
             "Ignored when --all is used.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Perform full exhaustive enumeration instead of random sampling "
             "(WARNING: slow, ~20 min for complexity 5).",
    )
    p.add_argument(
        "--max-complexity",
        type=int,
        default=5,
        metavar="C",
        help="Maximum total Wyckoff sites across all 3 elements (default: 5). "
             "Minimum is 3 (one per element).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="Random seed for reproducible sampling (default: None).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="W",
        help="Number of parallel workers for --all mode (default: all CPU cores).",
    )
    p.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Output file path. If omitted, prints to stdout.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.random_elements and args.elements:
        print("HELO HOW ARE YOU")
        print(args.elements)
        parser.error("--random-elements and positional ELEMENT arguments are mutually exclusive.")
    if not args.random_elements and not args.elements:
        parser.error("Provide element symbols or use --random-elements.")
    if args.random_elements and args.all:
        parser.error("--random-elements is not supported with --all (exhaustive enumeration requires fixed elements).")

    elements = args.elements if args.elements else None
    if elements is not None:
        try:
            _validate_elements(elements)
        except ValueError as e:
            parser.error(str(e))

    if args.max_complexity < 2:
        parser.error("--max-complexity must be at least 2.")

    if args.all:
        assert elements is not None  # guarded above: --random-elements+--all is rejected
        labels = enumerate_all(
            elements,
            max_complexity=args.max_complexity,
            n_workers=args.workers,
        )
    else:
        if args.n_samples <= 0:
            parser.error("--n-samples must be positive.")
        t0 = time.time()
        labels = random_sample(
            elements,
            n_samples=args.n_samples,
            max_complexity=args.max_complexity,
            seed=args.seed,
        )
        log.info(
            "Generated %d unique labels in %.2fs",
            len(labels),
            time.time() - t0,
        )

    output_text = "\n".join(labels) + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        log.info("Written to %s", args.output)
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()
