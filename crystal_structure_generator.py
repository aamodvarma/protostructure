#!/usr/bin/env python3
"""Generate crystal structures from protostructure labels.

Two-step API::

    template = parse_protostructure_label(label)
    structure = instantiate_template(template, lattice_params=..., wyckoff_params=...)

``wyckoff_params`` and ``lattice_params`` default to ``None``, which triggers
random (rejection-sampled) initialisation.
"""

from __future__ import annotations


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import product as iproduct
from typing import Optional

import numpy as np
from tqdm import tqdm
from pymatgen.analysis.prototypes import (
    CRYSTAL_FAMILY_SYMBOLS,
    WYCKOFF_MULTIPLICITY_DICT,
    WYCKOFF_POSITION_PARAM_DICT,
)
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class ProtoTemplate:
    """Parsed protostructure label."""

    sg_num: int
    pearson: str
    anon_formula: str
    elements: list[str]          # alphabetically sorted
    wyckoff_letters: list[list[str]]  # per-element list of Wyckoff letter strings

    @property
    def crystal_system(self) -> str:
        return SpaceGroup.from_int_number(self.sg_num).crystal_system

    @property
    def n_atoms(self) -> int:
        mult = WYCKOFF_MULTIPLICITY_DICT[str(self.sg_num)]
        return sum(mult[l] for letters in self.wyckoff_letters for l in letters)

    def atom_counts(self) -> list[int]:
        mult = WYCKOFF_MULTIPLICITY_DICT[str(self.sg_num)]
        return [sum(mult[l] for l in letters) for letters in self.wyckoff_letters]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_WYCKOFF_TOKEN = re.compile(r"(\d*)([a-z])")


def _parse_wyckoff_str(s: str) -> list[str]:
    """'2ab' → ['a','a','b'],  'dg' → ['d','g'],  'c' → ['c']."""
    letters: list[str] = []
    for m in _WYCKOFF_TOKEN.finditer(s):
        count = int(m.group(1)) if m.group(1) else 1
        letters.extend([m.group(2)] * count)
    return letters


def parse_protostructure_label(label: str) -> ProtoTemplate:
    """Parse an Aviary/AFLOW protostructure label.

    Format::

        {anon}_{pearson}_{sg_num}_{wyck_0}_{wyck_1}...:{el0}-{el1}-...

    Example::

        ABC_cF12_225_a_b_c:Cl-K-Na
    """
    aflow_part, chem_sys = label.rsplit(":", 1)
    elements = chem_sys.split("-")

    parts = aflow_part.split("_")
    if len(parts) < 3 + len(elements):
        raise ValueError(f"Malformed protostructure label: {label!r}")

    anon = parts[0]
    pearson = parts[1]
    sg_num = int(parts[2])
    wyckoff_parts = parts[3:]

    if len(wyckoff_parts) != len(elements):
        raise ValueError(
            f"Wyckoff parts ({len(wyckoff_parts)}) ≠ elements ({len(elements)}): {label!r}"
        )

    return ProtoTemplate(
        sg_num=sg_num,
        pearson=pearson,
        anon_formula=anon,
        elements=elements,
        wyckoff_letters=[_parse_wyckoff_str(wp) for wp in wyckoff_parts],
    )


# ---------------------------------------------------------------------------
# Wyckoff representative coordinates
# ---------------------------------------------------------------------------

def _test_lattice(sg_num: int) -> Lattice:
    """Minimal lattice with the correct crystal system for test structures."""
    family = CRYSTAL_FAMILY_SYMBOLS[SpaceGroup.from_int_number(sg_num).crystal_system]
    if family == "c":
        return Lattice.cubic(5.0)
    if family == "t":
        return Lattice.tetragonal(5.0, 7.0)
    if family == "h":
        return Lattice.hexagonal(5.0, 8.0)
    if family == "o":
        return Lattice.orthorhombic(5.0, 6.0, 7.0)
    if family == "m":
        return Lattice.monoclinic(5.0, 6.0, 7.0, 100.0)
    return Lattice.from_parameters(5.0, 6.0, 7.0, 80.0, 95.0, 105.0)


@lru_cache(maxsize=256)
def _special_wyckoff_coords(sg_num: int) -> dict[str, np.ndarray]:
    """Return {letter: frac_coord} for every n_free=0 Wyckoff site in sg_num.

    Uses rank-3 symmetry-operation fixed-point solver, then verifies assignments
    via SpacegroupAnalyzer.
    """
    sg = SpaceGroup.from_int_number(sg_num)
    ops = list(sg.symmetry_ops)
    mult_dict = WYCKOFF_MULTIPLICITY_DICT[str(sg_num)]
    param_dict = WYCKOFF_POSITION_PARAM_DICT[str(sg_num)]

    n_free0_letters = [l for l in mult_dict if param_dict[l] == 0]
    if not n_free0_letters:
        return {}

    # --- Collect candidate special points ---
    seen_orbit_keys: set = set()
    candidates: list[np.ndarray] = []

    def _try_add(c):
        x = np.array(c, float) % 1.0
        orbit = sg.get_orbit(x.tolist())
        key = frozenset(tuple(np.round(p, 3)) for p in orbit)
        if key not in seen_orbit_keys:
            seen_orbit_keys.add(key)
            candidates.append(x)

    for c in [
        (0, 0, 0), (0.5, 0.5, 0.5),
        (0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
        (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5),
        (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),
        (0.25, 0.75, 0.25), (0.75, 0.25, 0.25), (0.25, 0.25, 0.75),
        (0, 0.25, 0.25), (0.25, 0, 0.25), (0.25, 0.25, 0),
        (0.125, 0.125, 0.125), (0.375, 0.375, 0.375),
        (0, 0, 0.25), (0, 0.25, 0), (0.25, 0, 0),
        (0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5),
    ]:
        _try_add(c)

    # --- Rank-3 ops: unique fixed points ---
    for op in ops:
        R = op.rotation_matrix.astype(float)
        t = op.translation_vector
        A = R - np.eye(3)
        if np.linalg.matrix_rank(A, tol=1e-6) != 3:
            continue
        A_inv = np.linalg.inv(A)
        for n in iproduct(range(-2, 3), repeat=3):
            x = A_inv @ (-t + np.array(n, float))
            x = x % 1.0
            if np.all(x >= -1e-9) and np.all(x < 1.0 - 1e-9):
                _try_add(x)

    # --- Rank-2 op pairs: intersect two fixed lines → possible special point ---
    # Needed for hexagonal/trigonal sites like (1/3, 2/3, z).
    rank2_ops = [
        (op.rotation_matrix.astype(float), op.translation_vector)
        for op in ops
        if np.linalg.matrix_rank(op.rotation_matrix.astype(float) - np.eye(3), tol=1e-6) == 2
    ]
    for i, (R1, t1) in enumerate(rank2_ops):
        A1 = R1 - np.eye(3)
        for R2, t2 in rank2_ops[i + 1:]:
            A2 = R2 - np.eye(3)
            A_stacked = np.vstack([A1, A2])
            if np.linalg.matrix_rank(A_stacked, tol=1e-6) < 3:
                continue
            for n1 in iproduct(range(-1, 2), repeat=3):
                for n2 in iproduct(range(-1, 2), repeat=3):
                    b = np.concatenate([
                        -t1 + np.array(n1, float),
                        -t2 + np.array(n2, float),
                    ])
                    x, res, _, _ = np.linalg.lstsq(A_stacked, b, rcond=None)
                    if len(res) == 0 or res[0] > 1e-10:
                        if np.linalg.norm(A_stacked @ x - b) > 1e-6:
                            continue
                    x = x % 1.0
                    if np.all(x >= -1e-9) and np.all(x < 1.0 - 1e-9):
                        _try_add(x)

    # --- Filter to n_free=0 orbit sizes ---
    n0_mults = {mult_dict[l] for l in n_free0_letters}
    special = [(len(sg.get_orbit(c.tolist())), c) for c in candidates]
    special = [(m, c) for m, c in special if m in n0_mults]
    if not special:
        return {}

    # --- Build test structure and use SpacegroupAnalyzer to assign letters ---
    ELEMENTS = (
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca "
        "Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr"
    ).split()
    n = min(len(special), len(ELEMENTS))
    specs = ELEMENTS[:n]
    coords = [c for _, c in special[:n]]

    result: dict[str, np.ndarray] = {}
    try:
        struct = Structure.from_spacegroup(
            sg_num, _test_lattice(sg_num), specs, coords
        )
        sga = SpacegroupAnalyzer(struct, symprec=0.01)
        # Only trust assignments when the analyser detects the intended SG.
        # A wrong detection (higher symmetry) gives incorrect letter assignments.
        if sga.get_space_group_number() != sg_num:
            raise ValueError("wrong SG detected")
        sym = sga.get_symmetrized_structure()

        # equivalent_sites[i] = list of atoms in group i
        # wyckoff_symbols[i]   = Wyckoff label for group i (e.g. '4b')
        n_free0_set = set(n_free0_letters)
        for wyck_sym, group in zip(sym.wyckoff_symbols, sym.equivalent_sites):
            letter = wyck_sym[-1]
            if letter in result or letter not in n_free0_set:
                continue
            # Match any atom in this group back to an input candidate
            for atom in group:
                fc = atom.frac_coords % 1.0
                for c in coords:
                    if np.allclose(fc, np.array(c) % 1.0, atol=0.02):
                        result[letter] = np.array(c)
                        break
                if letter in result:
                    break
    except Exception:
        pass

    # Fallback: assign by orbit size + alphabetical letter order
    by_mult: dict[int, list[np.ndarray]] = {}
    for m, c in special:
        by_mult.setdefault(m, []).append(c)

    for letter in sorted(n_free0_letters, key=lambda l: (mult_dict[l], l)):
        if letter in result:
            continue
        m = mult_dict[letter]
        pool = by_mult.get(m, [])
        # skip candidates already used
        used = set(tuple(np.round(v, 4)) for v in result.values())
        for c in pool:
            if tuple(np.round(c, 4)) not in used:
                result[letter] = c
                break

    return result


def _wyckoff_parameterization(
    sg_num: int, letter: str
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return ``(base, free_dirs)`` for the given Wyckoff site.

    A valid representative fractional coordinate is::

        x = (base + s_0 * free_dirs[0] + s_1 * free_dirs[1] + ...) % 1

    where each ``s_i ∈ (0.05, 0.45)`` avoids accidental special positions.

    For n_free=0 sites ``free_dirs`` is empty and ``base`` is the exact position.
    """
    n_free = WYCKOFF_POSITION_PARAM_DICT[str(sg_num)][letter]
    mult = WYCKOFF_MULTIPLICITY_DICT[str(sg_num)][letter]
    sg = SpaceGroup.from_int_number(sg_num)
    ops = list(sg.symmetry_ops)

    if n_free == 0:
        table = _special_wyckoff_coords(sg_num)
        if letter in table:
            return table[letter], []
        raise ValueError(f"No special coord found for SG {sg_num} letter {letter!r}")

    if n_free == 3:
        return np.array([0.1, 0.2, 0.3]), [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

    # For n_free = 1 or 2: find a rank-(3-n_free) operation whose fixed-point
    # locus parametrises this Wyckoff site.
    target_rank = 3 - n_free
    for op in ops:
        R = op.rotation_matrix.astype(float)
        t = op.translation_vector
        A = R - np.eye(3)
        if np.linalg.matrix_rank(A, tol=1e-6) != target_rank:
            continue

        _, _, Vt = np.linalg.svd(A)
        null_dirs = [Vt[j] for j in range(target_rank, 3)]  # n_free direction(s)
        A_pinv = np.linalg.pinv(A)

        for n in iproduct(range(-1, 2), repeat=3):
            rhs = -t + np.array(n, float)
            x_part = A_pinv @ rhs
            if np.linalg.norm(A @ x_part - rhs) > 1e-6:
                continue  # not in range(A)

            # Test that a generic free-param value gives the right orbit size
            s_test = 0.17
            x_test = x_part.copy()
            for d in null_dirs:
                x_test = x_test + s_test * d
            x_test = x_test % 1.0
            if len(sg.get_orbit(x_test.tolist())) == mult:
                return x_part % 1.0, null_dirs

    raise ValueError(
        f"Cannot find Wyckoff parameterisation for SG {sg_num} letter {letter!r}"
    )


def _sample_wyckoff_coord(
    sg_num: int,
    letter: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a random valid fractional coordinate for the given Wyckoff site."""
    mult = WYCKOFF_MULTIPLICITY_DICT[str(sg_num)][letter]
    sg = SpaceGroup.from_int_number(sg_num)
    base, free_dirs = _wyckoff_parameterization(sg_num, letter)

    for _ in range(200):
        x = base.copy()
        for d in free_dirs:
            x = x + rng.uniform(0.05, 0.45) * d
        x = x % 1.0
        if len(sg.get_orbit(x.tolist())) == mult:
            return x

    raise RuntimeError(
        f"Failed to sample valid coord for SG {sg_num} letter {letter!r}"
    )


# ---------------------------------------------------------------------------
# Lattice helpers
# ---------------------------------------------------------------------------

def _build_lattice(sg_num: int, params: dict) -> Lattice:
    """Construct a pymatgen Lattice from explicit parameters.

    Only the parameters relevant to the crystal family are required:

    * Cubic:         ``a``
    * Tetragonal:    ``a``, ``c``
    * Hexagonal:     ``a``, ``c``
    * Orthorhombic:  ``a``, ``b``, ``c``
    * Monoclinic:    ``a``, ``b``, ``c``, ``beta``
    * Triclinic:     ``a``, ``b``, ``c``, ``alpha``, ``beta``, ``gamma``
    """
    family = CRYSTAL_FAMILY_SYMBOLS[SpaceGroup.from_int_number(sg_num).crystal_system]
    a = params["a"]
    if family == "c":
        return Lattice.cubic(a)
    if family == "t":
        return Lattice.tetragonal(a, params["c"])
    if family == "h":
        return Lattice.hexagonal(a, params["c"])
    if family == "o":
        return Lattice.orthorhombic(a, params["b"], params["c"])
    if family == "m":
        return Lattice.monoclinic(a, params["b"], params["c"], params["beta"])
    return Lattice.from_parameters(
        a, params["b"], params["c"],
        params["alpha"], params["beta"], params["gamma"],
    )


def _random_lattice(sg_num: int, rng: np.random.Generator, n_atoms: int = 1) -> Lattice:
    """Generate a random lattice consistent with the crystal system.

    Targets a total volume of 8–25 Å³/atom, back-computing ``a`` from
    the chosen axis ratios so that non-cubic families don't produce
    pathologically dilute cells.
    """
    family = CRYSTAL_FAMILY_SYMBOLS[SpaceGroup.from_int_number(sg_num).crystal_system]
    # Target volume keeps density in the physically realistic range regardless
    # of crystal system or axis ratios.
    target_vol = n_atoms * float(rng.uniform(8.0, 25.0))

    if family == "c":
        a = target_vol ** (1.0 / 3.0)
        return Lattice.cubic(a)
    if family == "t":
        rc = float(rng.uniform(0.7, 2.0))
        # V = a²c = a³·rc  →  a = (V/rc)^(1/3)
        a = (target_vol / rc) ** (1.0 / 3.0)
        return Lattice.tetragonal(a, a * rc)
    if family == "h":
        rc = float(rng.uniform(0.7, 2.5))
        # V = (√3/2)·a²c = a³·rc·√3/2
        a = (target_vol / (rc * np.sqrt(3) / 2.0)) ** (1.0 / 3.0)
        return Lattice.hexagonal(a, a * rc)
    if family == "o":
        rb = float(rng.uniform(0.7, 2.0))
        rc = float(rng.uniform(0.7, 2.0))
        # V = abc = a³·rb·rc
        a = (target_vol / (rb * rc)) ** (1.0 / 3.0)
        return Lattice.orthorhombic(a, a * rb, a * rc)
    if family == "m":
        rb = float(rng.uniform(0.7, 2.0))
        rc = float(rng.uniform(0.7, 2.0))
        beta = float(rng.uniform(90.0, 130.0))
        # V = abc·sin(β) = a³·rb·rc·sin(β)
        a = (target_vol / (rb * rc * np.sin(np.radians(beta)))) ** (1.0 / 3.0)
        return Lattice.monoclinic(a, a * rb, a * rc, beta)
    # Triclinic: V = abc·√(1−cos²α−cos²β−cos²γ+2cosα cosβ cosγ)
    rb = float(rng.uniform(0.7, 2.0))
    rc = float(rng.uniform(0.7, 2.0))
    for _ in range(100):
        alpha = float(rng.uniform(60.0, 120.0))
        beta  = float(rng.uniform(60.0, 120.0))
        gamma = float(rng.uniform(60.0, 120.0))
        ca, cb, cg = np.cos(np.radians(alpha)), np.cos(np.radians(beta)), np.cos(np.radians(gamma))
        det = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
        if det <= 0:
            continue
        a = (target_vol / (rb * rc * np.sqrt(det))) ** (1.0 / 3.0)
        try:
            lat = Lattice.from_parameters(a, a * rb, a * rc, alpha, beta, gamma)
            if lat.volume > 0.1:
                return lat
        except Exception:
            pass
    a = target_vol ** (1.0 / 3.0)
    return Lattice.from_parameters(a, a, a, 80.0, 95.0, 105.0)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

WyckoffParams = list[list[np.ndarray]]  # [elem_idx][orbit_idx] → frac_coords


def instantiate_template(
    template: ProtoTemplate,
    lattice_params: Optional[dict] = None,
    wyckoff_params: Optional[WyckoffParams] = None,
    min_dist: float = 1.5,
    max_attempts: int = 100,
    seed: Optional[int] = None,
) -> Structure:
    """Instantiate a :class:`ProtoTemplate` into a :class:`~pymatgen.core.Structure`.

    Parameters
    ----------
    template:
        Parsed protostructure template from :func:`parse_protostructure_label`.
    lattice_params:
        Dict of lattice parameters (``a``, ``b``, ``c``, ``alpha``, ``beta``,
        ``gamma``).  Only the keys relevant to the crystal family are needed.
        ``None`` → randomly chosen.
    wyckoff_params:
        ``wyckoff_params[i][j]`` = fractional coordinate of the representative
        atom for element ``i``'s ``j``-th Wyckoff orbit.  ``None`` → randomly
        sampled.
    min_dist:
        Minimum allowed interatomic distance in Å (rejection criterion).
    max_attempts:
        Number of random re-draws before raising.
    seed:
        Random seed.

    Returns
    -------
    Structure
    """
    rng = np.random.default_rng(seed)

    # --- Lattice ---
    if lattice_params is not None:
        lattice = _build_lattice(template.sg_num, lattice_params)
        fixed_lattice = True
    else:
        lattice = _random_lattice(template.sg_num, rng, n_atoms=template.n_atoms)
        fixed_lattice = False

    # --- Wyckoff coords ---
    fixed_wyckoff = wyckoff_params is not None

    def _make_wyckoff_params() -> WyckoffParams:
        return [
            [_sample_wyckoff_coord(template.sg_num, letter, rng) for letter in letters]
            for letters in template.wyckoff_letters
        ]

    if fixed_wyckoff:
        current_wyckoff = wyckoff_params
    else:
        current_wyckoff = _make_wyckoff_params()

    # --- Rejection sampling loop ---
    for attempt in range(max_attempts):
        # Flatten to (species, coord) pairs for from_spacegroup
        species: list[str] = []
        coords: list[np.ndarray] = []
        for elem, orb_coords in zip(template.elements, current_wyckoff):
            for coord in orb_coords:
                species.append(elem)
                coords.append(np.asarray(coord, dtype=float))

        try:
            struct = Structure.from_spacegroup(
                template.sg_num, lattice, species, coords
            )
        except Exception as exc:
            if fixed_lattice and fixed_wyckoff:
                raise RuntimeError(f"Structure.from_spacegroup failed: {exc}") from exc
            # Resample and retry
            if not fixed_wyckoff:
                current_wyckoff = _make_wyckoff_params()
            if not fixed_lattice:
                lattice = _random_lattice(template.sg_num, rng, n_atoms=template.n_atoms)
            continue

        # Verify atom count
        if len(struct) != template.n_atoms:
            if fixed_wyckoff:
                raise RuntimeError(
                    f"Generated {len(struct)} atoms but expected {template.n_atoms}. "
                    f"Possible duplicate n_free=0 Wyckoff orbits in label."
                )
            current_wyckoff = _make_wyckoff_params()
            continue

        # Check minimum interatomic distance
        dm = struct.distance_matrix.copy()
        np.fill_diagonal(dm, np.inf)
        if dm.min() >= min_dist:
            return struct

        # Too close: resample coords and/or scale lattice
        if not fixed_wyckoff:
            current_wyckoff = _make_wyckoff_params()
        if not fixed_lattice and attempt % 10 == 9:
            # Occasionally scale up the lattice
            lattice = Lattice(lattice.matrix * 1.1)

    raise RuntimeError(
        f"Could not build structure with min_dist={min_dist} Å "
        f"in {max_attempts} attempts for {template.pearson} SG {template.sg_num}."
    )







# x = parse_protostructure_label("A10B3C6_hP38_192_hl_ad_i:Hf-N-Zn")

with open("./labels.txt") as f:
    for line in tqdm(f, desc="Generating structures"):
        label = line.strip()
        print(f"Processing {label}...\n")
        try:
            template = parse_protostructure_label(label)
            structure = instantiate_template(template, seed=123)
        except Exception as exc:
            print(f"  → Failed to generate structure: {exc}\n")
            continue
        cif_filename = f"./cifs/{template.anon_formula}_{template.pearson}_{template.sg_num}.cif"
        CifWriter(structure).write_file(cif_filename)
        print(f"  → Wrote {cif_filename}\n")