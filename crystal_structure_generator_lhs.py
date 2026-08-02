#!/usr/bin/env python3
"""Generate crystal structures from protostructure labels via LHS + distance optimisation.

Two-step API::

    template = parse_protostructure_label(label)
    structure = instantiate_template_lhs(template, n_samples=20)

``instantiate_template_lhs`` performs Latin Hypercube Sampling over all
symmetry-allowed degrees of freedom (lattice parameters + Wyckoff free
parameters) and then refines the best candidate by minimising the interatomic
distance penalty from arXiv:2601.21393.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import multiprocessing as mp
import hashlib
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from math import floor
from typing import Mapping, Optional

import numpy as np
from tqdm import tqdm
from pymatgen.analysis.prototypes import (
    CRYSTAL_FAMILY_SYMBOLS,
    WYCKOFF_MULTIPLICITY_DICT,
    WYCKOFF_POSITION_PARAM_DICT,
)
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.groups import SpaceGroup
from pyxtal.symmetry import Group as PyXtalGroup

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
    if not s:
        raise ValueError("A Wyckoff section cannot be empty.")
    letters: list[str] = []
    end = 0
    for m in _WYCKOFF_TOKEN.finditer(s):
        if m.start() != end:
            raise ValueError(f"Malformed Wyckoff section: {s!r}")
        count = int(m.group(1)) if m.group(1) else 1
        if count < 1:
            raise ValueError(f"Wyckoff repetition must be positive: {s!r}")
        letters.extend([m.group(2)] * count)
        end = m.end()
    if end != len(s):
        raise ValueError(f"Malformed Wyckoff section: {s!r}")
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

    if not 1 <= sg_num <= 230:
        raise ValueError(f"Space-group number must be in 1..230: {sg_num}")

    if len(wyckoff_parts) != len(elements):
        raise ValueError(
            f"Wyckoff parts ({len(wyckoff_parts)}) ≠ elements ({len(elements)}): {label!r}"
        )

    for symbol in elements:
        try:
            parsed = Element(symbol)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid element symbol {symbol!r}.") from exc
        if parsed.symbol != symbol:
            raise ValueError(
                f"Element symbol must be {parsed.symbol!r}, not {symbol!r}."
            )
    if len(set(elements)) != len(elements):
        raise ValueError(f"Elements must be distinct: {elements}")

    parsed_wyckoffs = [_parse_wyckoff_str(wp) for wp in wyckoff_parts]
    valid_letters = WYCKOFF_MULTIPLICITY_DICT[str(sg_num)]
    counts: dict[str, int] = {}
    for letter in (letter for group in parsed_wyckoffs for letter in group):
        if letter not in valid_letters:
            raise ValueError(f"Unknown Wyckoff letter {letter!r} for SG {sg_num}.")
        counts[letter] = counts.get(letter, 0) + 1
    free_params = WYCKOFF_POSITION_PARAM_DICT[str(sg_num)]
    repeated_fixed = [
        letter
        for letter, count in counts.items()
        if count > 1 and free_params[letter] == 0
    ]
    if repeated_fixed:
        raise ValueError(
            f"Fixed Wyckoff positions cannot be repeated in SG {sg_num}: "
            f"{sorted(repeated_fixed)}"
        )

    template = ProtoTemplate(
        sg_num=sg_num,
        pearson=pearson,
        anon_formula=anon,
        elements=elements,
        wyckoff_letters=parsed_wyckoffs,
    )
    pearson_match = re.fullmatch(r"[a-z][A-Z](\d+)", pearson)
    if pearson_match is None or int(pearson_match.group(1)) != template.n_atoms:
        raise ValueError(
            f"Pearson symbol {pearson!r} is inconsistent with {template.n_atoms} atoms."
        )
    return template


# ---------------------------------------------------------------------------
# Wyckoff representative coordinates (shared helpers)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2048)
def _wyckoff_parameterization(sg_num: int, letter: str):
    """Return PyXtal's exact affine parameterization for one Wyckoff letter.

    Multiplicity and free-parameter count do not uniquely identify a Wyckoff
    letter. PyXtal supplies the tabulated affine expressions, avoiding the
    ambiguous orbit inference previously used here.
    """
    expected_mult = WYCKOFF_MULTIPLICITY_DICT[str(sg_num)][letter]
    expected_dof = WYCKOFF_POSITION_PARAM_DICT[str(sg_num)][letter]
    for wp in PyXtalGroup(sg_num).Wyckoff_positions:
        if wp.letter != letter:
            continue
        if wp.multiplicity != expected_mult or int(wp.get_dof()) != expected_dof:
            raise ValueError(
                f"Inconsistent Wyckoff data for SG {sg_num} letter {letter!r}."
            )
        return wp
    raise ValueError(f"Unknown Wyckoff position SG {sg_num} letter {letter!r}.")


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


def estimate_target_volume(
    template: ProtoTemplate, packing_fraction: float = 0.55
) -> float:
    """Estimate a unit-cell volume from composition when no model value is supplied.

    The paper uses a fitted effective-radius regression whose coefficients are
    not published. This deterministic radius/packing estimate is therefore a
    fallback for synthetic-data generation, not a reproduction of that model.
    Pass ``target_volume`` to :func:`instantiate_template_lhs` when a trained
    volume-model prediction is available.
    """
    if not 0.0 < packing_fraction <= 1.0:
        raise ValueError("packing_fraction must lie in (0, 1].")

    sphere_volume = 0.0
    for symbol, count in zip(template.elements, template.atom_counts()):
        element = Element(symbol)
        radius = element.atomic_radius_calculated or element.atomic_radius
        if radius is None:
            radius = 1.5
        sphere_volume += count * (4.0 * np.pi / 3.0) * float(radius) ** 3
    return sphere_volume / packing_fraction


def _axis_ratios(values: np.ndarray, max_axis_ratio: float) -> np.ndarray:
    """Map unit values to positive ratios with max(length)/min(length) bounded."""
    if max_axis_ratio < 1.0:
        raise ValueError("max_axis_ratio must be at least 1.")
    logs = (2.0 * np.asarray(values, dtype=float) - 1.0) * np.log(max_axis_ratio)
    all_logs = np.concatenate(([0.0], logs))
    spread = float(np.ptp(all_logs))
    if spread > np.log(max_axis_ratio):
        all_logs *= np.log(max_axis_ratio) / spread
    return np.exp(all_logs)


def _lattice_from_unit(
    u: np.ndarray,
    family: str,
    target_volume: float,
    max_axis_ratio: float,
) -> Lattice:
    """Build a Lattice from a unit [0, 1] DOF vector.

    DOF layout (all values in [0, 1]):
      cubic:       [vol_frac]
      tetragonal:  [vol_frac, c_ratio]
      hexagonal:   [vol_frac, c_ratio]
      ortho:       [vol_frac, b_ratio, c_ratio]
      monoclinic:  [vol_frac, b_ratio, c_ratio, beta_frac]
      triclinic:   [vol_frac, b_ratio, c_ratio, alpha_frac, beta_frac, gamma_frac]

    The first DOF samples volume within ±10% of ``target_volume``. Length
    ratios are sampled in log space and constrained so that the longest axis
    is at most ``max_axis_ratio`` times the shortest (14 in the paper).
    """
    target_vol = target_volume * (0.9 + 0.2 * float(u[0]))

    if family == "c":
        a = target_vol ** (1.0 / 3.0)
        return Lattice.cubic(a)
    if family == "t":
        _, rc = _axis_ratios(u[1:2], max_axis_ratio)
        # V = a²c = a³·rc  →  a = (V/rc)^(1/3)
        a = (target_vol / rc) ** (1.0 / 3.0)
        return Lattice.tetragonal(a, a * rc)
    if family == "h":
        _, rc = _axis_ratios(u[1:2], max_axis_ratio)
        # V = (√3/2)·a²c = a³·rc·√3/2
        a = (target_vol / (rc * np.sqrt(3) / 2.0)) ** (1.0 / 3.0)
        return Lattice.hexagonal(a, a * rc)
    if family == "o":
        _, rb, rc = _axis_ratios(u[1:3], max_axis_ratio)
        # V = abc = a³·rb·rc
        a = (target_vol / (rb * rc)) ** (1.0 / 3.0)
        return Lattice.orthorhombic(a, a * rb, a * rc)
    if family == "m":
        _, rb, rc = _axis_ratios(u[1:3], max_axis_ratio)
        beta = 60.0 + float(u[3]) * 60.0
        # V = abc·sin(β) = a³·rb·rc·sin(β)
        a = (target_vol / (rb * rc * np.sin(np.radians(beta)))) ** (1.0 / 3.0)
        return Lattice.monoclinic(a, a * rb, a * rc, beta)
    # Triclinic: V = abc·√(1−cos²α−cos²β−cos²γ+2cosα cosβ cosγ)
    _, rb, rc = _axis_ratios(u[1:3], max_axis_ratio)
    alpha = 60.0 + float(u[3]) * 60.0
    beta = 60.0 + float(u[4]) * 60.0
    gamma = 60.0 + float(u[5]) * 60.0
    ca, cb, cg = (
        np.cos(np.radians(alpha)),
        np.cos(np.radians(beta)),
        np.cos(np.radians(gamma)),
    )
    det = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
    # Some independently sampled angle triples do not define a positive cell.
    # Contract only such triples toward 90 degrees until they do.
    while det <= 1e-4:
        alpha, beta, gamma = [90.0 + 0.9 * (x - 90.0) for x in (alpha, beta, gamma)]
        ca, cb, cg = np.cos(np.radians([alpha, beta, gamma]))
        det = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
    a = (target_vol / (rb * rc * np.sqrt(det))) ** (1.0 / 3.0)
    return Lattice.from_parameters(a, a * rb, a * rc, alpha, beta, gamma)


def _wyckoff_coord_from_unit(
    wyckoff_position,
    u: np.ndarray,
) -> np.ndarray:
    """Convert unit [0, 1] values to a Wyckoff representative fractional coord.

    The paper samples each free coordinate over (0, 1]. LHS itself samples the
    open interior of [0, 1], so no additional narrowing is applied.
    """
    return (
        np.asarray(
            wyckoff_position.get_position_from_free_xyzs(np.asarray(u, dtype=float)),
            dtype=float,
        )
        % 1.0
    )


# Per-element list of PyXtal Wyckoff-position objects, one per occupied orbit.
_ParamData = list[list[object]]


def _build_structure_from_unit_vec(
    u: np.ndarray,
    template: ProtoTemplate,
    family: str,
    target_volume: float,
    max_axis_ratio: float,
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
        lattice = _lattice_from_unit(
            u[:n_lat_dofs], family, target_volume, max_axis_ratio
        )

    species: list[str] = []
    coords: list[np.ndarray] = []
    idx = n_lat_dofs

    for elem, letters, orbit_params in zip(
        template.elements, template.wyckoff_letters, param_data
    ):
        for _letter, wyckoff_position in zip(letters, orbit_params):
            n_free = int(wyckoff_position.get_dof())
            coord = _wyckoff_coord_from_unit(wyckoff_position, u[idx : idx + n_free])
            idx += n_free
            orbit = np.asarray(wyckoff_position.apply_ops(coord), dtype=float) % 1.0
            if len(orbit) != wyckoff_position.multiplicity:
                return None
            species.extend([elem] * len(orbit))
            coords.extend(orbit)

    try:
        # Expand with the same tabulated Wyckoff operators that generated the
        # representatives. Mixing symmetry tables can introduce setting/origin
        # mismatches for some space groups.
        struct = Structure(lattice, species, coords)
    except Exception:
        return None

    if len(struct) != template.n_atoms:
        return None

    return struct


def _shortest_lattice_translation(lattice: Lattice) -> float:
    """Return the shortest nonzero translation needed for self distances."""
    # A shortest vector cannot be longer than the shortest basis vector. Let
    # pymatgen enumerate every periodic image within that finite sphere; unlike
    # a fixed [-k, k]^3 search this remains correct for highly skewed cells.
    _frac, distances, _indices, _images = lattice.get_points_in_sphere(
        [[0.0, 0.0, 0.0]],
        [0.0, 0.0, 0.0],
        min(lattice.abc) + 1e-8,
        zip_results=False,
    )
    nonzero = distances[distances > 1e-8]
    if len(nonzero) == 0:
        raise ValueError("Could not identify a nonzero lattice translation.")
    return float(np.min(nonzero))


@lru_cache(maxsize=4096)
def _default_pair_distance(symbol_a: str, symbol_b: str) -> float:
    """Fallback pair cutoff based on tabulated atomic radii."""
    radii = []
    for symbol in (symbol_a, symbol_b):
        element = Element(symbol)
        radius = element.atomic_radius_calculated or element.atomic_radius or 1.0
        radii.append(float(radius))
    # Deliberately conservative: this is a clash-removal threshold, not an
    # equilibrium bond-length predictor.
    return 0.40 * sum(radii)


def _minimum_distance_matrix(
    struct: Structure,
    min_dist: float | None,
    pair_min_distances: Mapping[tuple[str, str], float] | None,
) -> np.ndarray:
    symbols = [site.specie.symbol for site in struct]
    thresholds = np.empty((len(struct), len(struct)), dtype=float)
    overrides = pair_min_distances or {}
    for i, symbol_i in enumerate(symbols):
        for j, symbol_j in enumerate(symbols):
            key = tuple(sorted((symbol_i, symbol_j)))
            if key in overrides:
                threshold = float(overrides[key])
            elif min_dist is not None:
                threshold = float(min_dist)
            else:
                threshold = _default_pair_distance(symbol_i, symbol_j)
            if threshold <= 0:
                raise ValueError(f"Minimum distance for {key} must be positive.")
            thresholds[i, j] = threshold
    return thresholds


def _distance_penalty(
    struct: Structure,
    min_dist: float | None = None,
    pair_min_distances: Mapping[tuple[str, str], float] | None = None,
) -> float:
    """Evaluate Eq. (1) of arXiv:2601.21393.

    Off-diagonal entries use shortest periodic atom--atom distances. Diagonal
    entries use the shortest nonzero lattice translation, preventing the zero
    self-distance artifact described in the paper.
    """
    distances = struct.distance_matrix.copy()
    np.fill_diagonal(distances, _shortest_lattice_translation(struct.lattice))
    thresholds = _minimum_distance_matrix(struct, min_dist, pair_min_distances)
    return 0.5 * float(np.maximum(0.0, thresholds - distances).sum())


def _minimum_distance_margin(
    struct: Structure,
    min_dist: float | None,
    pair_min_distances: Mapping[tuple[str, str], float] | None,
) -> float:
    distances = struct.distance_matrix.copy()
    np.fill_diagonal(distances, _shortest_lattice_translation(struct.lattice))
    thresholds = _minimum_distance_matrix(struct, min_dist, pair_min_distances)
    return float(np.min(distances - thresholds))


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


def recommended_sample_count(n_dofs: int, n_atoms: int) -> int:
    """Paper heuristic for the number of initial structures.

    Flooring the result preserves the stated 3000-atom budget.
    """
    if n_dofs < 0 or n_atoms < 1:
        raise ValueError("n_dofs must be nonnegative and n_atoms positive.")
    count = min(
        20.0 + 10.0 * n_dofs**1.3 / n_atoms**0.5,
        3000.0 / n_atoms,
    )
    return max(1, floor(count))


def instantiate_template_lhs(
    template: ProtoTemplate,
    n_samples: int | None = None,
    lattice_params: Optional[dict] = None,
    min_dist: float | None = None,
    pair_min_distances: Mapping[tuple[str, str], float] | None = None,
    target_volume: float | None = None,
    max_axis_ratio: float = 14.0,
    optimize: bool = True,
    seed: Optional[int] = None,
) -> Structure:
    """Instantiate a :class:`ProtoTemplate` via LHS + distance optimisation.

    Algorithm
    ---------
    1. Identify all symmetry-allowed degrees of freedom:

       * Lattice parameters encoded in [0, 1] (unless ``lattice_params`` is
         given, in which case the lattice is fixed).
       * One free parameter per symmetry-allowed Wyckoff direction over (0, 1).

    2. Draw ``n_samples`` points from a Latin Hypercube over the full DOF
       space, ensuring uniform stratified coverage.

    3. Build a Structure for each sample and evaluate the paper's distance
       penalty. Keep the lowest-penalty sample as the optimisation start.

    4. If ``optimize=True``, refine with L-BFGS-B minimisation of the
       repulsion energy (bounded to [0, 1] for each DOF).

    5. Return the structure if it satisfies ``min_dist``; otherwise raise.

    Parameters
    ----------
    template:
        Parsed protostructure template.
    n_samples:
        Number of LHS candidates. ``None`` uses the paper's DOF/atom heuristic.
    lattice_params:
        If given, fix the lattice (same format as ``_build_lattice``).
        ``None`` → lattice DOFs are included in the LHS/optimisation.
    min_dist:
        Optional scalar minimum distance (Å). If omitted, element-pair
        cutoffs are estimated from atomic radii. ``pair_min_distances`` can
        override selected sorted element pairs, e.g. ``{("Li", "O"): 1.6}``.
    target_volume:
        Predicted unit-cell volume in Å³. LHS samples ±10% around it. If
        omitted, a documented radius/packing fallback is used because the
        paper's fitted volume-model coefficients are not published.
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
            orbit_params.append(_wyckoff_parameterization(template.sg_num, letter))
        param_data.append(orbit_params)

    # Determine DOF counts
    fixed_lattice: Optional[Lattice] = None
    if lattice_params is not None:
        fixed_lattice = _build_lattice(template.sg_num, lattice_params)
        n_lat_dofs = 0
        sampled_target_volume = fixed_lattice.volume
    else:
        n_lat_dofs = _n_lattice_dofs(family)
        sampled_target_volume = (
            estimate_target_volume(template) if target_volume is None else target_volume
        )
        if sampled_target_volume <= 0:
            raise ValueError("target_volume must be positive.")

    n_wyck_dofs = sum(
        int(wyckoff_position.get_dof())
        for orbit_params in param_data
        for wyckoff_position in orbit_params
    )
    n_dofs = n_lat_dofs + n_wyck_dofs
    if n_samples is None:
        n_samples = recommended_sample_count(n_dofs, n_atoms)
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    # Fully-determined structure: no free parameters
    if n_dofs == 0:
        u0 = np.zeros(0)
        struct = _build_structure_from_unit_vec(
            u0,
            template,
            family,
            sampled_target_volume,
            max_axis_ratio,
            param_data,
            fixed_lattice,
            n_lat_dofs,
        )
        if (
            struct is not None
            and _distance_penalty(struct, min_dist, pair_min_distances) <= 1e-10
        ):
            return struct
        raise RuntimeError(
            f"Failed to build fully-determined structure for "
            f"{template.pearson} SG {template.sg_num}."
        )

    # Objective closure for scipy
    def objective(u: np.ndarray) -> float:
        struct = _build_structure_from_unit_vec(
            u,
            template,
            family,
            sampled_target_volume,
            max_axis_ratio,
            param_data,
            fixed_lattice,
            n_lat_dofs,
        )
        return (
            1e9
            if struct is None
            else _distance_penalty(struct, min_dist, pair_min_distances)
        )

    # --- Phase 1: LHS screening ---
    lhs_points = _lhs_sample(n_samples, n_dofs, rng)

    best_energy = np.inf
    best_struct: Optional[Structure] = None
    starts: list[tuple[float, np.ndarray, Structure]] = []

    for u in lhs_points:
        struct = _build_structure_from_unit_vec(
            u,
            template,
            family,
            sampled_target_volume,
            max_axis_ratio,
            param_data,
            fixed_lattice,
            n_lat_dofs,
        )
        if struct is None:
            continue
        energy = _distance_penalty(struct, min_dist, pair_min_distances)
        starts.append((energy, u.copy(), struct))
        if energy < best_energy:
            best_energy = energy
            best_struct = struct

    if best_struct is None:
        raise RuntimeError(
            f"All {n_samples} LHS samples failed for "
            f"{template.pearson} SG {template.sg_num}."
        )

    margin = _minimum_distance_margin(best_struct, min_dist, pair_min_distances)
    if margin >= -1e-8:
        return best_struct

    if not optimize:
        raise RuntimeError(
            f"Best LHS sample has a minimum-distance margin of {margin:.3f} Å "
            f"and optimize=False for {template.pearson} SG {template.sg_num}."
        )

    # --- Phase 2: multi-start distance optimisation ---
    bounds = [(0.0, 1.0)] * n_dofs
    starts.sort(key=lambda item: item[0])
    candidates: list[Structure] = []
    # A hinge penalty can have stationary points. Optimizing several of the
    # best-separated LHS starts is substantially more reliable than one start.
    for _energy, start_u, start_struct in starts[: min(2, len(starts))]:
        candidates.append(start_struct)
        result = scipy_minimize(
            objective,
            start_u,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 150, "ftol": 1e-8, "gtol": 1e-6},
        )
        opt_struct = _build_structure_from_unit_vec(
            result.x,
            template,
            family,
            sampled_target_volume,
            max_axis_ratio,
            param_data,
            fixed_lattice,
            n_lat_dofs,
        )
        if opt_struct is not None:
            candidates.append(opt_struct)

    valid_candidates = [
        candidate
        for candidate in candidates
        if _minimum_distance_margin(candidate, min_dist, pair_min_distances) >= -1e-8
    ]
    if valid_candidates:
        return max(
            valid_candidates,
            key=lambda candidate: _minimum_distance_margin(
                candidate, min_dist, pair_min_distances
            ),
        )

    best_margin = max(
        _minimum_distance_margin(candidate, min_dist, pair_min_distances)
        for candidate in candidates
    )

    raise RuntimeError(
        f"Could not satisfy minimum-distance constraints "
        f"after LHS ({n_samples} samples) + L-BFGS-B optimisation "
        f"for {template.pearson} SG {template.sg_num}; best margin "
        f"was {best_margin:.3f} Å."
    )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def _process_label(
    label: str,
    out_dir: str,
    seed: int = 123,
    n_samples: int | None = None,
    min_dist: float | None = None,
    max_axis_ratio: float = 14.0,
    target_volume_per_atom: float | None = None,
) -> str:
    """Worker function: parse, instantiate, and write one CIF. Returns a status message."""
    try:
        template = parse_protostructure_label(label)
        label_hash = hashlib.sha256(label.encode()).hexdigest()
        label_seed = seed ^ int(label_hash[:16], 16)
        target_volume = (
            None
            if target_volume_per_atom is None
            else target_volume_per_atom * template.n_atoms
        )
        structure = instantiate_template_lhs(
            template,
            n_samples=n_samples,
            min_dist=min_dist,
            target_volume=target_volume,
            max_axis_ratio=max_axis_ratio,
            seed=label_seed,
        )
    except Exception as exc:
        return f"[FAIL] {label}: {exc}"
    cif_filename = os.path.join(
        out_dir,
        f"{template.anon_formula}_{template.pearson}_{template.sg_num}_"
        f"{label_hash[:12]}_lhs.cif",
    )
    CifWriter(structure).write_file(cif_filename)
    return f"[OK]   {label} → {cif_filename}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate crystal structures from protostructure labels"
    )
    parser.add_argument(
        "--input",
        default="./labels/labels-mixed-100k-volfixed.txt",
        help="Input labels file",
    )
    parser.add_argument(
        "--outdir", default="./100k-volfixed", help="Output directory for CIF files"
    )
    parser.add_argument(
        "--workers", type=int, default=mp.cpu_count(), help="Number of worker processes"
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="LHS samples per protostructure (default: paper DOF/atom heuristic)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=None,
        help="Scalar minimum distance in Å (default: element-pair radius estimate)",
    )
    parser.add_argument(
        "--max-axis-ratio",
        type=float,
        default=14.0,
        help="Maximum longest/shortest lattice-axis ratio (default: 14)",
    )
    parser.add_argument(
        "--target-volume-per-atom",
        type=float,
        default=None,
        help="Optional volume-model estimate in Å³/atom; sampled within ±10%%",
    )
    args = parser.parse_args()

    if args.n_samples is not None and args.n_samples < 1:
        parser.error("--n-samples must be positive.")
    if args.min_dist is not None and args.min_dist <= 0:
        parser.error("--min-dist must be positive.")
    if args.max_axis_ratio < 1:
        parser.error("--max-axis-ratio must be at least 1.")
    if args.target_volume_per_atom is not None and args.target_volume_per_atom <= 0:
        parser.error("--target-volume-per-atom must be positive.")

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.input) as f:
        labels = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(labels)} labels with {args.workers} workers...")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _process_label,
                label,
                args.outdir,
                args.seed,
                args.n_samples,
                args.min_dist,
                args.max_axis_ratio,
                args.target_volume_per_atom,
            ): label
            for label in labels
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Generating structures"
        ):
            print(future.result())
