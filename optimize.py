#!/usr/bin/env python3
"""High-throughput batched MLIP relaxation for generated crystal structures.

This script is intended for large directories of CIF files. It launches one
persistent TorchSim/MACE worker per GPU, dynamically batches independent
structures, relaxes atomic positions and (by default) cells with FIRE, and
writes restartable output chunks in extended XYZ format.

Typical usage on four GPUs::

    python optimize.py \
        --input crystal-structures-200k \
        --output relaxed-mpa0 \
        --gpus 0,1,2,3 \
        --chunk-size 2000 \
        --fmax 0.03 \
        --max-steps 300

The output is split into ``shard-*/chunk-*.extxyz`` files. Each chunk also has
JSONL metadata and a ``.done.json`` checkpoint. Re-running the same command
resumes at unfinished chunks. Use a new output directory when changing model
or relaxation settings.

The batched Frechet-cell workflow does not explicitly project onto the initial
space-group degrees of freedom. Exact Wyckoff-symmetry preservation requires a
separate symmetry-constrained optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing generated structure files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for relaxed shards and metadata.",
    )
    parser.add_argument(
        "--pattern",
        default="*.cif",
        help="Recursive input glob relative to --input (default: *.cif).",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Comma-separated physical GPU IDs (default: 0,1,2,3).",
    )
    parser.add_argument(
        "--model",
        default="medium-mpa-0",
        help="MACE foundation-model name or local checkpoint path "
        "(default: medium-mpa-0 / MACE-MPA-0).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Input structures checkpointed together per worker (default: 2000).",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.03,
        help="Maximum force convergence threshold in eV/Angstrom (default: 0.03).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum FIRE steps per structure (default: 300).",
    )
    parser.add_argument(
        "--steps-between-swaps",
        type=int,
        default=5,
        help="FIRE steps between convergence checks/hot swaps (default: 5).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Model precision (default: float32; float64 is slower but stricter).",
    )
    parser.add_argument(
        "--no-cueq",
        action="store_true",
        help="Disable NVIDIA cuEquivariance acceleration.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default=None,
        help="Optional torch.compile mode. Benchmark before enabling.",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=512,
        help="Per-atom capacity for the fast NVIDIA neighbor list (default: 512). "
        "Overflows automatically fall back to TorchSim's uncapped linked-cell list.",
    )
    parser.add_argument(
        "--positions-only",
        action="store_true",
        help="Relax atomic positions while keeping the generated cell fixed.",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.5,
        help="Post-relaxation minimum periodic distance in Angstrom (default: 0.5).",
    )
    parser.add_argument(
        "--only-converged",
        action="store_true",
        help="Do not write valid structures that reached max-steps unconverged.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed used to shuffle and distribute inputs (default: 123).",
    )
    return parser


def _parse_gpu_ids(value: str) -> list[str]:
    gpu_ids = [part.strip() for part in value.split(",") if part.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU ID.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"GPU IDs must be unique: {gpu_ids}")
    if any(not gpu_id.isdigit() for gpu_id in gpu_ids):
        raise ValueError(f"GPU IDs must be nonnegative integers: {gpu_ids}")
    return gpu_ids


def _validate_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> list[str]:
    if sys.version_info < (3, 12):
        parser.error(
            "This batched TorchSim workflow requires Python 3.12 or newer; "
            f"the active interpreter is Python {sys.version.split()[0]}. "
            "Create a clean Python 3.12 environment before installing "
            "torch-sim-atomistic, mace-torch, and cuEquivariance."
        )
    try:
        gpu_ids = _parse_gpu_ids(args.gpus)
    except ValueError as exc:
        parser.error(str(exc))
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive.")
    if args.fmax <= 0:
        parser.error("--fmax must be positive.")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive.")
    if args.steps_between_swaps < 1:
        parser.error("--steps-between-swaps must be positive.")
    if args.steps_between_swaps > args.max_steps:
        parser.error("--steps-between-swaps cannot exceed --max-steps.")
    if args.max_neighbors < 1:
        parser.error("--max-neighbors must be positive.")
    if args.min_distance <= 0:
        parser.error("--min-distance must be positive.")
    if not args.input.is_dir():
        parser.error(f"--input must be an existing directory: {args.input}")
    return gpu_ids


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def _run_config(args: argparse.Namespace, n_workers: int) -> dict[str, Any]:
    """Settings that must not change when resuming an output directory."""
    return {
        "input": str(args.input.resolve()),
        "pattern": args.pattern,
        "model": args.model,
        "n_workers": n_workers,
        "chunk_size": args.chunk_size,
        "fmax": args.fmax,
        "max_steps": args.max_steps,
        "steps_between_swaps": args.steps_between_swaps,
        "dtype": args.dtype,
        "cueq": not args.no_cueq,
        "compile_mode": args.compile_mode,
        "max_neighbors": args.max_neighbors,
        "positions_only": args.positions_only,
        "min_distance": args.min_distance,
        "only_converged": args.only_converged,
        "seed": args.seed,
    }


def _prepare_manifests(args: argparse.Namespace, n_workers: int) -> list[list[str]]:
    """Create stable shuffled worker assignments, or load them on resume."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run-config.json"
    requested_config = _run_config(args, n_workers)

    if config_path.exists():
        existing_config = json.loads(config_path.read_text())
        # Neighbor capacity changes allocation only; completed calculations that did
        # not overflow remain equivalent, so older runs are safe to resume.
        if "max_neighbors" not in existing_config:
            existing_config["max_neighbors"] = args.max_neighbors
            _atomic_write_text(
                config_path,
                json.dumps(existing_config, indent=2, sort_keys=True) + "\n",
            )
        if existing_config != requested_config:
            raise RuntimeError(
                "The output directory contains a run with different settings. "
                "Use the original settings or choose a new --output directory.\n"
                f"Existing: {json.dumps(existing_config, sort_keys=True)}\n"
                f"Requested: {json.dumps(requested_config, sort_keys=True)}"
            )
    else:
        _atomic_write_text(
            config_path, json.dumps(requested_config, indent=2, sort_keys=True) + "\n"
        )

    manifest_dir = output / "manifests"
    manifest_paths = [manifest_dir / f"shard-{idx:02d}.txt" for idx in range(n_workers)]
    present = [path.exists() for path in manifest_paths]
    if any(present) and not all(present):
        raise RuntimeError(
            "Only some shard manifests exist. Use a new --output directory or "
            "restore the missing manifests."
        )

    if all(present):
        assignments = [
            [line for line in path.read_text().splitlines() if line]
            for path in manifest_paths
        ]
        missing = [
            path for shard in assignments for path in shard if not Path(path).is_file()
        ]
        if missing:
            preview = "\n".join(missing[:10])
            raise RuntimeError(
                f"Manifest input files are missing, for example:\n{preview}"
            )
        return assignments

    files = sorted(
        path.resolve() for path in args.input.rglob(args.pattern) if path.is_file()
    )
    if not files:
        raise RuntimeError(
            f"No input files matched {args.pattern!r} below {args.input.resolve()}."
        )
    random.Random(args.seed).shuffle(files)
    assignments = [[] for _ in range(n_workers)]
    for index, path in enumerate(files):
        assignments[index % n_workers].append(str(path))

    manifest_dir.mkdir(parents=True, exist_ok=True)
    for path, shard in zip(manifest_paths, assignments):
        _atomic_write_text(path, "\n".join(shard) + "\n")
    return assignments


def _checkpoint_model(model: str) -> str:
    """Download/cache a foundation checkpoint once before GPU workers start."""
    try:
        from mace.calculators.foundations_models import download_mace_mp_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            "MACE is not installed. Install it with: pip install mace-torch"
        ) from exc
    return str(Path(download_mace_mp_checkpoint(model)).resolve())


def _structure_id(path: str, input_root: str) -> str:
    try:
        relative = str(Path(path).resolve().relative_to(Path(input_root).resolve()))
    except ValueError:
        relative = str(Path(path).resolve())
    digest = hashlib.sha256(relative.encode()).hexdigest()[:16]
    return f"{Path(path).stem}-{digest}"


def _minimum_periodic_distance(atoms: Any, np: Any) -> float:
    """Minimum atom-atom or nonzero self-image distance."""
    cell = np.asarray(atoms.cell.array, dtype=float)
    coeffs = np.array(
        [
            (i, j, k)
            for i in range(-2, 3)
            for j in range(-2, 3)
            for k in range(-2, 3)
            if (i, j, k) != (0, 0, 0)
        ],
        dtype=float,
    )
    shortest_translation = float(np.min(np.linalg.norm(coeffs @ cell, axis=1)))
    if len(atoms) < 2:
        return shortest_translation
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
    np.fill_diagonal(distances, np.inf)
    return min(shortest_translation, float(np.min(distances)))


def _load_chunk(
    paths: list[str], input_root: str
) -> tuple[list[tuple[str, str, Any]], list[dict]]:
    from ase.io import read

    loaded: list[tuple[str, str, Any]] = []
    failures: list[dict] = []
    for path in paths:
        structure_id = _structure_id(path, input_root)
        try:
            atoms = read(path, index=-1)
            if len(atoms) < 1:
                raise ValueError("structure contains no atoms")
            if not bool(atoms.pbc.all()):
                atoms.pbc = True
            atoms.calc = None
            loaded.append((path, structure_id, atoms))
        except Exception as exc:  # isolate malformed input files
            failures.append(
                {
                    "id": structure_id,
                    "input": path,
                    "status": "read_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return loaded, failures


def _state_to_results(
    state: Any,
    items: list[tuple[str, str, Any]],
    convergence_fn: Any,
    args: argparse.Namespace,
    torch: Any,
    np: Any,
) -> tuple[list[Any], list[dict]]:
    atoms_list = state.to_atoms()
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if len(atoms_list) != len(items):
        raise RuntimeError(
            f"TorchSim returned {len(atoms_list)} systems for {len(items)} inputs."
        )

    energies = state.energy.detach().cpu().numpy()
    forces = state.forces.detach().cpu().numpy()
    stresses = getattr(state, "stress", None)
    if stresses is not None:
        stresses = stresses.detach().cpu().numpy()
    converged = convergence_fn(state, None).detach().cpu().numpy().astype(bool)

    output_atoms: list[Any] = []
    metadata: list[dict] = []
    atom_offset = 0
    for index, (atoms, (path, structure_id, _original)) in enumerate(
        zip(atoms_list, items)
    ):
        n_atoms = len(atoms)
        atom_forces = np.asarray(
            forces[atom_offset : atom_offset + n_atoms], dtype=float
        )
        atom_offset += n_atoms
        energy = float(energies[index])
        max_force = float(np.linalg.norm(atom_forces, axis=1).max())
        min_distance = _minimum_periodic_distance(atoms, np)
        volume = float(atoms.get_volume())
        finite = bool(
            np.isfinite(energy)
            and np.isfinite(atom_forces).all()
            and np.isfinite(atoms.positions).all()
            and np.isfinite(atoms.cell.array).all()
        )
        valid = finite and volume > 1e-6 and min_distance >= args.min_distance
        is_converged = bool(converged[index])

        if not valid:
            status = "invalid"
        elif is_converged:
            status = "converged"
        else:
            status = "max_steps"

        record = {
            "id": structure_id,
            "input": path,
            "status": status,
            "converged": is_converged,
            "valid": valid,
            "n_atoms": n_atoms,
            "energy_eV": energy,
            "energy_per_atom_eV": energy / n_atoms,
            "max_force_eV_per_A": max_force,
            "volume_A3": volume,
            "min_distance_A": min_distance,
        }
        if stresses is not None:
            record["stress_eV_per_A3"] = (
                np.asarray(stresses[index]).reshape(-1).tolist()
            )
        metadata.append(record)

        should_write = valid and (is_converged or not args.only_converged)
        if should_write:
            atoms.calc = None
            atoms.info.update(
                {
                    "structure_id": structure_id,
                    "source_file": path,
                    "mlip_status": status,
                    "mlip_converged": is_converged,
                    "mlip_energy_eV": energy,
                    "mlip_energy_per_atom_eV": energy / n_atoms,
                    "mlip_max_force_eV_per_A": max_force,
                    "mlip_min_distance_A": min_distance,
                }
            )
            atoms.arrays["mlip_forces"] = atom_forces
            if stresses is not None:
                atoms.info["mlip_stress_eV_per_A3"] = np.asarray(
                    stresses[index]
                ).reshape(-1)
            output_atoms.append(atoms)

    if atom_offset != len(forces):
        raise RuntimeError(
            f"Consumed {atom_offset} force rows but TorchSim returned {len(forces)}."
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_atoms, metadata


def _relax_with_isolation(
    items: list[tuple[str, str, Any]],
    model: Any,
    convergence_fn: Any,
    args: argparse.Namespace,
    ts: Any,
    torch: Any,
    np: Any,
) -> tuple[list[Any], list[dict]]:
    """Relax a batch, recursively isolating any pathological structure."""
    if not items:
        return [], []
    try:
        init_kwargs = (
            {} if args.positions_only else {"cell_filter": ts.CellFilter.frechet}
        )
        state = ts.optimize(
            system=[item[2] for item in items],
            model=model,
            optimizer=ts.Optimizer.fire,
            convergence_fn=convergence_fn,
            max_steps=args.max_steps,
            steps_between_swaps=args.steps_between_swaps,
            autobatcher=True,
            init_kwargs=init_kwargs,
            trajectory_reporter=None,
            pbar=False,
        )
        return _state_to_results(state, items, convergence_fn, args, torch=torch, np=np)
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(items) == 1:
            path, structure_id, _atoms = items[0]
            return [], [
                {
                    "id": structure_id,
                    "input": path,
                    "status": "relax_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ]
        midpoint = len(items) // 2
        print(
            f"Batch of {len(items)} failed ({type(exc).__name__}: {exc}); "
            "retrying its halves.",
            flush=True,
        )
        left_atoms, left_meta = _relax_with_isolation(
            items[:midpoint], model, convergence_fn, args, ts, torch, np
        )
        right_atoms, right_meta = _relax_with_isolation(
            items[midpoint:], model, convergence_fn, args, ts, torch, np
        )
        return left_atoms + right_atoms, left_meta + right_meta


def _write_chunk(
    shard_dir: Path,
    chunk_index: int,
    atoms_list: list[Any],
    metadata: list[dict],
    elapsed_seconds: float,
) -> None:
    from ase.io import write

    stem = f"chunk-{chunk_index:06d}"
    xyz_path = shard_dir / f"{stem}.extxyz"
    metadata_path = shard_dir / f"{stem}.jsonl"
    done_path = shard_dir / f"{stem}.done.json"
    xyz_temp = shard_dir / f".{stem}.extxyz.tmp-{os.getpid()}"
    metadata_temp = shard_dir / f".{stem}.jsonl.tmp-{os.getpid()}"

    if atoms_list:
        write(xyz_temp, atoms_list, format="extxyz")
    else:
        xyz_temp.write_text("")
    with metadata_temp.open("w") as handle:
        for record in metadata:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(xyz_temp, xyz_path)
    os.replace(metadata_temp, metadata_path)

    counts: dict[str, int] = {}
    for record in metadata:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    done = {
        "chunk": chunk_index,
        "input_count": len(metadata),
        "written_count": len(atoms_list),
        "status_counts": counts,
        "elapsed_seconds": elapsed_seconds,
    }
    _atomic_write_text(done_path, json.dumps(done, indent=2, sort_keys=True) + "\n")


def _make_neighbor_list(max_neighbors: int) -> Any:
    """Build a fast bounded neighbor list with an uncapped overflow fallback."""
    import torch
    from nvalchemiops.neighbors.neighbor_utils import NeighborOverflowError
    from nvalchemiops.torch.neighbors import batch_naive_neighbor_list
    from torch_sim.neighbors import torch_nl_linked_cell
    from torch_sim.neighbors.utils import normalize_inputs

    warned_about_overflow = False

    def neighbor_list(
        positions: Any,
        cell: Any,
        pbc: Any,
        cutoff: float,
        system_idx: Any,
        self_interaction: bool = False,
    ) -> tuple[Any, Any, Any]:
        nonlocal warned_about_overflow
        n_systems = int(system_idx.max().item()) + 1
        normalized_cell, normalized_pbc = normalize_inputs(cell, pbc, n_systems)
        try:
            result = batch_naive_neighbor_list(
                positions=positions,
                cutoff=cutoff,
                batch_idx=system_idx.to(torch.int32),
                cell=normalized_cell,
                pbc=normalized_pbc.to(torch.bool),
                max_neighbors=max_neighbors,
                return_neighbor_list=True,
            )
        except NeighborOverflowError as exc:
            if not warned_about_overflow:
                print(
                    "Neighbor capacity exceeded "
                    f"({exc.num_neighbors} > {exc.max_neighbors}); using the "
                    "uncapped TorchSim linked-cell neighbor list for dense states.",
                    flush=True,
                )
                warned_about_overflow = True
            return torch_nl_linked_cell(
                positions,
                normalized_cell,
                normalized_pbc,
                cutoff,
                system_idx,
                self_interaction,
            )

        mapping = result[0].to(dtype=torch.long)
        if len(result) == 3:
            shifts_idx = result[2].to(dtype=normalized_cell.dtype)
        else:
            shifts_idx = torch.zeros(
                (mapping.shape[1], 3),
                dtype=normalized_cell.dtype,
                device=positions.device,
            )
        system_mapping = system_idx[mapping[0]]

        if self_interaction:
            n_atoms = positions.shape[0]
            atom_indices = torch.arange(
                n_atoms, device=positions.device, dtype=torch.long
            )
            mapping = torch.cat(
                (mapping, torch.stack((atom_indices, atom_indices))), dim=1
            )
            shifts_idx = torch.cat(
                (
                    shifts_idx,
                    torch.zeros(
                        (n_atoms, 3),
                        dtype=normalized_cell.dtype,
                        device=positions.device,
                    ),
                ),
                dim=0,
            )
            system_mapping = torch.cat((system_mapping, system_idx), dim=0)

        return mapping, system_mapping, shifts_idx

    return neighbor_list


def _worker(
    worker_index: int,
    physical_gpu: str,
    paths: list[str],
    model_path: str,
    args: argparse.Namespace,
) -> None:
    # CUDA_VISIBLE_DEVICES must be set before importing torch in the spawned worker.
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    try:
        import numpy as np
        import torch
        import torch_sim as ts
        from mace.calculators.mace_torchsim import MaceTorchSimModel

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Worker {worker_index} cannot access CUDA GPU {physical_gpu}."
            )
        torch.cuda.set_device(0)
        if args.dtype == "float32":
            torch.set_float32_matmul_precision("high")
        dtype = torch.float32 if args.dtype == "float32" else torch.float64

        print(
            f"[shard {worker_index}] loading {args.model} on physical GPU "
            f"{physical_gpu} ({torch.cuda.get_device_name(0)})",
            flush=True,
        )
        model = MaceTorchSimModel(
            model=model_path,
            device=torch.device("cuda"),
            dtype=dtype,
            neighbor_list_fn=_make_neighbor_list(args.max_neighbors),
            compute_forces=True,
            compute_stress=not args.positions_only,
            enable_cueq=not args.no_cueq,
            compile_mode=args.compile_mode,
        )
        convergence_fn = ts.generate_force_convergence_fn(
            force_tol=args.fmax,
            include_cell_forces=not args.positions_only,
        )

        shard_dir = args.output.resolve() / f"shard-{worker_index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        n_chunks = (len(paths) + args.chunk_size - 1) // args.chunk_size
        for chunk_index in range(n_chunks):
            done_path = shard_dir / f"chunk-{chunk_index:06d}.done.json"
            if done_path.exists():
                continue
            start = chunk_index * args.chunk_size
            chunk_paths = paths[start : start + args.chunk_size]
            started = time.monotonic()
            loaded, read_failures = _load_chunk(
                chunk_paths, input_root=str(args.input.resolve())
            )
            relaxed_atoms, relaxation_metadata = _relax_with_isolation(
                loaded, model, convergence_fn, args, ts, torch, np
            )
            metadata = read_failures + relaxation_metadata
            # Metadata must account for every input before checkpointing.
            if len(metadata) != len(chunk_paths):
                raise RuntimeError(
                    f"Chunk {chunk_index} has {len(chunk_paths)} inputs but "
                    f"{len(metadata)} metadata records."
                )
            elapsed = time.monotonic() - started
            _write_chunk(
                shard_dir,
                chunk_index,
                relaxed_atoms,
                metadata,
                elapsed_seconds=elapsed,
            )
            print(
                f"[shard {worker_index}] chunk {chunk_index + 1}/{n_chunks}: "
                f"{len(chunk_paths)} inputs, {len(relaxed_atoms)} written, "
                f"{elapsed:.1f} s",
                flush=True,
            )
    except Exception:
        traceback.print_exc()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    gpu_ids = _validate_args(args, parser)

    try:
        assignments = _prepare_manifests(args, len(gpu_ids))
        total = sum(map(len, assignments))
        print(
            f"Prepared {total:,} structures across {len(gpu_ids)} GPU workers "
            f"({', '.join(f'{len(shard):,}' for shard in assignments)} per shard).",
            flush=True,
        )
        model_path = _checkpoint_model(args.model)
        print(f"Using cached model checkpoint: {model_path}", flush=True)
    except Exception as exc:
        parser.error(str(exc))

    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for worker_index, (physical_gpu, paths) in enumerate(zip(gpu_ids, assignments)):
        process = context.Process(
            target=_worker,
            args=(worker_index, physical_gpu, paths, model_path, args),
            name=f"relax-gpu-{physical_gpu}",
        )
        process.start()
        processes.append(process)

    failed = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed.append((process.name, process.exitcode))
    if failed:
        print(f"Workers failed: {failed}", file=sys.stderr)
        return 1
    print(f"Relaxation complete. Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
