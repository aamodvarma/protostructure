#!/usr/bin/env python3
"""Generate crystal structures from protostructure labels via LHS + distance optimisation.

Two-step API::

    template = parse_protostructure_label(label)
    structure = instantiate_template_lhs(template, n_samples=20)

``instantiate_template_lhs`` performs Latin Hypercube Sampling over all
symmetry-allowed degrees of freedom (lattice parameters + Wyckoff free
parameters) and then refines the best candidate with L-BFGS-B minimisation of
a soft pairwise-repulsion energy.
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
    elements: list[str]  # alphabetically sorted
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
# Wyckoff representative coordinates (shared helpers)
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
    """Return {letter: frac_coord} for every n_free=0 Wyckoff site in sg_num."""
    sg = SpaceGroup.from_int_number(sg_num)
    ops = list(sg.symmetry_ops)
    mult_dict = WYCKOFF_MULTIPLICITY_DICT[str(sg_num)]
    param_dict = WYCKOFF_POSITION_PARAM_DICT[str(sg_num)]

    n_free0_letters = [l for l in mult_dict if param_dict[l] == 0]
    if not n_free0_letters:
        return {}

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
        (0, 0, 0),
        (0.5, 0.5, 0.5),
        (0.25, 0.25, 0.25),
        (0.75, 0.75, 0.75),
        (0.5, 0, 0),
        (0, 0.5, 0),
        (0, 0, 0.5),
        (0.5, 0.5, 0),
        (0.5, 0, 0.5),
        (0, 0.5, 0.5),
        (0.25, 0.75, 0.25),
        (0.75, 0.25, 0.25),
        (0.25, 0.25, 0.75),
        (0, 0.25, 0.25),
        (0.25, 0, 0.25),
        (0.25, 0.25, 0),
        (0.125, 0.125, 0.125),
        (0.375, 0.375, 0.375),
        (0, 0, 0.25),
        (0, 0.25, 0),
        (0.25, 0, 0),
        (0.5, 0.25, 0.25),
        (0.25, 0.5, 0.25),
        (0.25, 0.25, 0.5),
    ]:
        _try_add(c)

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

    rank2_ops = [
        (op.rotation_matrix.astype(float), op.translation_vector)
        for op in ops
        if np.linalg.matrix_rank(op.rotation_matrix.astype(float) - np.eye(3), tol=1e-6)
        == 2
    ]
    for i, (R1, t1) in enumerate(rank2_ops):
        A1 = R1 - np.eye(3)
        for R2, t2 in rank2_ops[i + 1 :]:
            A2 = R2 - np.eye(3)
            A_stacked = np.vstack([A1, A2])
            if np.linalg.matrix_rank(A_stacked, tol=1e-6) < 3:
                continue
            for n1 in iproduct(range(-1, 2), repeat=3):
                for n2 in iproduct(range(-1, 2), repeat=3):
                    b = np.concatenate(
                        [
                            -t1 + np.array(n1, float),
                            -t2 + np.array(n2, float),
                        ]
                    )
                    x, res, _, _ = np.linalg.lstsq(A_stacked, b, rcond=None)
                    if len(res) == 0 or res[0] > 1e-10:
                        if np.linalg.norm(A_stacked @ x - b) > 1e-6:
                            continue
                    x = x % 1.0
                    if np.all(x >= -1e-9) and np.all(x < 1.0 - 1e-9):
                        _try_add(x)

    n0_mults = {mult_dict[l] for l in n_free0_letters}
    special = [(len(sg.get_orbit(c.tolist())), c) for c in candidates]
    special = [(m, c) for m, c in special if m in n0_mults]
    if not special:
        return {}

    ELEMENTS = (
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca "
        "Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr"
    ).split()
    n = min(len(special), len(ELEMENTS))
    specs = ELEMENTS[:n]
    coords = [c for _, c in special[:n]]

    result: dict[str, np.ndarray] = {}
    try:
        struct = Structure.from_spacegroup(sg_num, _test_lattice(sg_num), specs, coords)
        sga = SpacegroupAnalyzer(struct, symprec=0.01)
        if sga.get_space_group_number() != sg_num:
            raise ValueError("wrong SG detected")
        sym = sga.get_symmetrized_structure()
        n_free0_set = set(n_free0_letters)
        for wyck_sym, group in zip(sym.wyckoff_symbols, sym.equivalent_sites):
            letter = wyck_sym[-1]
            if letter in result or letter not in n_free0_set:
                continue
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

    by_mult: dict[int, list[np.ndarray]] = {}
    for m, c in special:
        by_mult.setdefault(m, []).append(c)

    for letter in sorted(n_free0_letters, key=lambda l: (mult_dict[l], l)):
        if letter in result:
            continue
        m = mult_dict[letter]
        pool = by_mult.get(m, [])
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

    target_rank = 3 - n_free
    for op in ops:
        R = op.rotation_matrix.astype(float)
        t = op.translation_vector
        A = R - np.eye(3)
        if np.linalg.matrix_rank(A, tol=1e-6) != target_rank:
            continue

        _, _, Vt = np.linalg.svd(A)
        null_dirs = [Vt[j] for j in range(target_rank, 3)]
        A_pinv = np.linalg.pinv(A)

        for n in iproduct(range(-1, 2), repeat=3):
            rhs = -t + np.array(n, float)
            x_part = A_pinv @ rhs
            if np.linalg.norm(A @ x_part - rhs) > 1e-6:
                continue

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


# ---------------------------------------------------------------------------
# Lattice helpers
# ---------------------------------------------------------------------------


def _build_lattice(sg_num: int, params: dict) -> Lattice:
    """Construct a pymatgen Lattice from explicit parameters."""
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
        a,
        params["b"],
        params["c"],
        params["alpha"],
        params["beta"],
        params["gamma"],
    )


# ---------------------------------------------------------------------------
# LHS helpers
# ---------------------------------------------------------------------------


def _n_lattice_dofs(family: str) -> int:
    """Number of independent lattice parameters for the given crystal family."""
    return {"c": 1, "t": 2, "h": 2, "o": 3, "m": 4}.get(family, 6)


def _lattice_from_unit(u: np.ndarray, family: str, n_atoms: int) -> Lattice:
    """Build a Lattice from a unit [0, 1] DOF vector.

    DOF layout (all values in [0, 1]):
      cubic:       [vol_frac]
      tetragonal:  [vol_frac, c_ratio]
      hexagonal:   [vol_frac, c_ratio]
      ortho:       [vol_frac, b_ratio, c_ratio]
      monoclinic:  [vol_frac, b_ratio, c_ratio, beta_frac]
      triclinic:   [vol_frac, b_ratio, c_ratio, alpha_frac, beta_frac, gamma_frac]

    target_vol = n_atoms * (8 + vol_frac * 17)   → 8–25 Å³/atom
    ``a`` is back-computed from target_vol and the axis ratios so that
    non-cubic systems cannot produce pathologically dilute cells when
    ratios are large.
    axis ratio = 0.7 + ratio_frac * 1.3          → [0.7, 2.0]
    c/a (hex)  = 0.7 + c_frac * 1.8              → [0.7, 2.5]
    beta (mono) = 90 + beta_frac * 40            → [90°, 130°]
    angles (tri) = 60 + angle_frac * 60          → [60°, 120°]
    """
    target_vol = n_atoms * (8.0 + float(u[0]) * 17.0)

    def ratio(v: float) -> float:
        return 0.7 + float(v) * 1.3

    if family == "c":
        a = target_vol ** (1.0 / 3.0)
        return Lattice.cubic(a)
    if family == "t":
        rc = ratio(u[1])
        # V = a²c = a³·rc  →  a = (V/rc)^(1/3)
        a = (target_vol / rc) ** (1.0 / 3.0)
        return Lattice.tetragonal(a, a * rc)
    if family == "h":
        rc = 0.7 + float(u[1]) * 1.8
        # V = (√3/2)·a²c = a³·rc·√3/2
        a = (target_vol / (rc * np.sqrt(3) / 2.0)) ** (1.0 / 3.0)
        return Lattice.hexagonal(a, a * rc)
    if family == "o":
        rb, rc = ratio(u[1]), ratio(u[2])
        # V = abc = a³·rb·rc
        a = (target_vol / (rb * rc)) ** (1.0 / 3.0)
        return Lattice.orthorhombic(a, a * rb, a * rc)
    if family == "m":
        rb, rc = ratio(u[1]), ratio(u[2])
        beta = 90.0 + float(u[3]) * 40.0
        # V = abc·sin(β) = a³·rb·rc·sin(β)
        a = (target_vol / (rb * rc * np.sin(np.radians(beta)))) ** (1.0 / 3.0)
        return Lattice.monoclinic(a, a * rb, a * rc, beta)
    # Triclinic: V = abc·√(1−cos²α−cos²β−cos²γ+2cosα cosβ cosγ)
    rb = ratio(u[1])
    rc = ratio(u[2])
    alpha = 60.0 + float(u[3]) * 60.0
    beta = 60.0 + float(u[4]) * 60.0
    gamma = 60.0 + float(u[5]) * 60.0
    ca, cb, cg = (
        np.cos(np.radians(alpha)),
        np.cos(np.radians(beta)),
        np.cos(np.radians(gamma)),
    )
    det = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
    if det > 0:
        a = (target_vol / (rb * rc * np.sqrt(det))) ** (1.0 / 3.0)
        try:
            lat = Lattice.from_parameters(a, a * rb, a * rc, alpha, beta, gamma)
            if lat.volume > 0.1:
                return lat
        except Exception:
            pass
    a = target_vol ** (1.0 / 3.0)
    return Lattice.from_parameters(a, a, a, 80.0, 95.0, 105.0)


def _wyckoff_coord_from_unit(
    base: np.ndarray,
    free_dirs: list[np.ndarray],
    u: np.ndarray,
) -> np.ndarray:
    """Convert unit [0, 1] values to a Wyckoff representative fractional coord.

    Each u_i ∈ [0, 1] is mapped to s_i ∈ [0.05, 0.45] to stay away from
    accidental special positions at the boundaries.
    """
    x = base.copy()
    for d, ui in zip(free_dirs, u):
        x = x + (0.05 + float(ui) * 0.40) * d
    return x % 1.0


# param_data type: per-element list of (base, free_dirs) tuples, one per orbit
_ParamData = list[list[tuple[np.ndarray, list[np.ndarray]]]]


def _build_structure_from_unit_vec(
    u: np.ndarray,
    template: ProtoTemplate,
    family: str,
    n_atoms: int,
    param_data: _ParamData,
    fixed_lattice: Optional[Lattice],
    n_lat_dofs: int,
) -> Optional[Structure]:
    """Build a Structure from the packed unit DOF vector.

    Layout of ``u``:
      u[0 : n_lat_dofs]       → lattice DOFs (ignored when fixed_lattice given)
      u[n_lat_dofs : ...]     → Wyckoff free parameters, one per free direction
                                 in traversal order (element → orbit → direction)

    Returns ``None`` on any failure (bad lattice, spacegroup mismatch, etc.).
    """
    if fixed_lattice is not None:
        lattice = fixed_lattice
    else:
        lattice = _lattice_from_unit(u[:n_lat_dofs], family, n_atoms)

    species: list[str] = []
    coords: list[np.ndarray] = []
    idx = n_lat_dofs

    for elem, letters, orbit_params in zip(
        template.elements, template.wyckoff_letters, param_data
    ):
        for _letter, (base, free_dirs) in zip(letters, orbit_params):
            n_free = len(free_dirs)
            coord = _wyckoff_coord_from_unit(base, free_dirs, u[idx : idx + n_free])
            idx += n_free
            species.append(elem)
            coords.append(coord)

    try:
        struct = Structure.from_spacegroup(template.sg_num, lattice, species, coords)
    except Exception:
        return None

    if len(struct) != template.n_atoms:
        return None

    return struct


def _soft_repulsion(struct: Structure, d_scale: float = 2.5) -> float:
    """Smooth pairwise repulsive energy ∑ (d_scale / d_ij)^6.

    This is differentiable (unlike min-distance) and suitable as an
    L-BFGS-B objective.  Minimising it pushes all atom pairs apart.
    The cutoff at 3·d_scale avoids evaluating long-range pairs.
    """
    dm = struct.distance_matrix.copy()
    np.fill_diagonal(dm, np.inf)
    close = dm[dm < d_scale * 3.0]
    if len(close) == 0:
        return 0.0
    return float(np.sum((d_scale / close) ** 6))


def _lhs_sample(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Return an (n_samples, n_dims) Latin Hypercube sample in [0, 1].

    Uses scipy.stats.qmc.LatinHypercube when available; falls back to a
    manual stratified permutation sampler.
    """
    try:
        from scipy.stats.qmc import LatinHypercube

        sampler = LatinHypercube(d=n_dims, seed=int(rng.integers(0, 2**31)))
        return sampler.random(n=n_samples)
    except ImportError:
        result = np.zeros((n_samples, n_dims))
        for j in range(n_dims):
            perm = rng.permutation(n_samples)
            result[:, j] = (perm + rng.uniform(size=n_samples)) / n_samples
        return result


# ---------------------------------------------------------------------------
# Main LHS API
# ---------------------------------------------------------------------------


def instantiate_template_lhs(
    template: ProtoTemplate,
    n_samples: int = 20,
    lattice_params: Optional[dict] = None,
    min_dist: float = 1.5,
    optimize: bool = True,
    seed: Optional[int] = None,
) -> Structure:
    """Instantiate a :class:`ProtoTemplate` via LHS + distance optimisation.

    Algorithm
    ---------
    1. Identify all symmetry-allowed degrees of freedom:

       * Lattice parameters encoded in [0, 1] (unless ``lattice_params`` is
         given, in which case the lattice is fixed).
       * One free parameter per symmetry-allowed Wyckoff direction, encoded in
         [0, 1] → s ∈ [0.05, 0.45].

    2. Draw ``n_samples`` points from a Latin Hypercube over the full DOF
       space, ensuring uniform stratified coverage.

    3. Build a Structure for each sample; compute the soft pairwise-repulsion
       energy ∑ (d_scale / d_ij)^6.  Keep the sample with the lowest energy
       (most spread-out atoms) as the starting point.

    4. If ``optimize=True``, refine with L-BFGS-B minimisation of the
       repulsion energy (bounded to [0, 1] for each DOF).

    5. Return the structure if it satisfies ``min_dist``; otherwise raise.

    Parameters
    ----------
    template:
        Parsed protostructure template.
    n_samples:
        Number of LHS candidates to evaluate before optimisation.
    lattice_params:
        If given, fix the lattice (same format as ``_build_lattice``).
        ``None`` → lattice DOFs are included in the LHS/optimisation.
    min_dist:
        Minimum interatomic distance (Å) required in the returned structure.
    optimize:
        Whether to run L-BFGS-B refinement after LHS screening.
    seed:
        Random seed.

    Returns
    -------
    Structure
    """
    from scipy.optimize import minimize as scipy_minimize

    rng = np.random.default_rng(seed)
    sg = SpaceGroup.from_int_number(template.sg_num)
    family = CRYSTAL_FAMILY_SYMBOLS[sg.crystal_system]
    n_atoms = template.n_atoms

    # Pre-compute Wyckoff parameterizations (cached under the hood)
    param_data: _ParamData = []
    for letters in template.wyckoff_letters:
        orbit_params = []
        for letter in letters:
            base, free_dirs = _wyckoff_parameterization(template.sg_num, letter)
            orbit_params.append((base, free_dirs))
        param_data.append(orbit_params)

    # Determine DOF counts
    fixed_lattice: Optional[Lattice] = None
    if lattice_params is not None:
        fixed_lattice = _build_lattice(template.sg_num, lattice_params)
        n_lat_dofs = 0
    else:
        n_lat_dofs = _n_lattice_dofs(family)

    n_wyck_dofs = sum(
        len(free_dirs)
        for orbit_params in param_data
        for (_base, free_dirs) in orbit_params
    )
    n_dofs = n_lat_dofs + n_wyck_dofs

    # Fully-determined structure: no free parameters
    if n_dofs == 0:
        u0 = np.zeros(0)
        struct = _build_structure_from_unit_vec(
            u0, template, family, n_atoms, param_data, fixed_lattice, n_lat_dofs
        )
        if struct is not None:
            return struct
        raise RuntimeError(
            f"Failed to build fully-determined structure for "
            f"{template.pearson} SG {template.sg_num}."
        )

    # Objective closure for scipy
    def objective(u: np.ndarray) -> float:
        struct = _build_structure_from_unit_vec(
            u, template, family, n_atoms, param_data, fixed_lattice, n_lat_dofs
        )
        return 1e9 if struct is None else _soft_repulsion(struct)

    # --- Phase 1: LHS screening ---
    lhs_points = _lhs_sample(n_samples, n_dofs, rng)

    best_energy = np.inf
    best_struct: Optional[Structure] = None
    best_u: Optional[np.ndarray] = None

    for u in lhs_points:
        struct = _build_structure_from_unit_vec(
            u, template, family, n_atoms, param_data, fixed_lattice, n_lat_dofs
        )
        if struct is None:
            continue
        energy = _soft_repulsion(struct)
        if energy < best_energy:
            best_energy = energy
            best_struct = struct
            best_u = u.copy()

    if best_struct is None:
        raise RuntimeError(
            f"All {n_samples} LHS samples failed for "
            f"{template.pearson} SG {template.sg_num}."
        )

    if not optimize:
        dm = best_struct.distance_matrix.copy()
        np.fill_diagonal(dm, np.inf)
        if dm.min() >= min_dist:
            return best_struct
        raise RuntimeError(
            f"Best LHS sample has min_dist={dm.min():.3f} Å < {min_dist} Å "
            f"and optimize=False for {template.pearson} SG {template.sg_num}."
        )

    # --- Phase 2: L-BFGS-B optimisation from best LHS point ---
    bounds = [(0.0, 1.0)] * n_dofs
    result = scipy_minimize(
        objective,
        best_u,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-8, "gtol": 1e-6},
    )

    opt_struct = _build_structure_from_unit_vec(
        result.x, template, family, n_atoms, param_data, fixed_lattice, n_lat_dofs
    )

    # Prefer the optimised structure if it satisfies min_dist
    for candidate in (opt_struct, best_struct):
        if candidate is None:
            continue
        dm = candidate.distance_matrix.copy()
        np.fill_diagonal(dm, np.inf)
        if dm.min() >= min_dist:
            return candidate

    raise RuntimeError(
        f"Could not build structure with min_dist={min_dist} Å "
        f"after LHS ({n_samples} samples) + L-BFGS-B optimisation "
        f"for {template.pearson} SG {template.sg_num}."
    )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open("./labels/labels-mixed-100k-volfixed.txt") as f:
        for line in tqdm(f, desc="Generating structures (LHS)"):
            label = line.strip()
            print(f"Processing {label}...\n")
            try:
                template = parse_protostructure_label(label)
                structure = instantiate_template_lhs(template, n_samples=20, seed=123)
            except Exception as exc:
                print(f"  → Failed: {exc}\n")
                continue
            cif_filename = (
                f"./100k-volfixed/{template.anon_formula}_{template.pearson}"
                f"_{template.sg_num}_lhs.cif"
            )
            CifWriter(structure).write_file(cif_filename)
            print(f"  → Wrote {cif_filename}\n")
