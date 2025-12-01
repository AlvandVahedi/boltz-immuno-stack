from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

ALIGN_CHAIN_PREFIXES = {"A", "B"}
RMSD_CHAIN_PREFIXES = {"D", "C"}

def compute_masks(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    chains = data["chains"]
    chain_mask = data["mask"].astype(bool)

    num_atoms = int(sum(chain["atom_num"] for chain in chains))
    alignment_mask = np.zeros(num_atoms, dtype=bool)
    rmsd_mask = np.zeros(num_atoms, dtype=bool)

    for idx, chain in enumerate(chains):
        if idx >= len(chain_mask) or not chain_mask[idx]:
            continue

        chain_name = _get_chain_name(chain)
        prefix = _first_alpha(chain_name)

        start = int(chain["atom_idx"])
        end = start + int(chain["atom_num"])

        if prefix in ALIGN_CHAIN_PREFIXES:
            print(f"======== Found alignment chain: {chain_name} ========")
            alignment_mask[start:end] = True
        elif prefix in RMSD_CHAIN_PREFIXES:
            print(f"======== Found RMSD chain: {chain_name} ========")
            rmsd_mask[start:end] = True

    return alignment_mask, rmsd_mask

def process_structure(npz_path: Path, dry_run: bool = False, print_residues: bool = False) -> None:
    with np.load(npz_path) as data:
        alignment_mask, rmsd_mask = compute_masks(data)

        if dry_run:
            print(
                f"{npz_path.name}: align atoms={alignment_mask.sum()} "
                f"(len={alignment_mask.shape[0]}), "
                f"rmsd atoms={rmsd_mask.sum()} (len={rmsd_mask.shape[0]})"
            )
            if print_residues:
                _describe_mask(npz_path.stem, data, alignment_mask, "ALIGN")
                _describe_mask(npz_path.stem, data, rmsd_mask, "RMSD")
            return

        updated = {key: data[key] for key in data.files}
        updated["alignment_mask"] = alignment_mask
        updated["rmsd_mask"] = rmsd_mask

    tmp_path = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **updated)
    tmp_path.replace(npz_path)


def _to_str(value: str | bytes | np.str_) -> str:
    if isinstance(value, bytes):
        value = value.decode()
    return str(value)


def _describe_mask(
    record_id: str,
    data: np.lib.npyio.NpzFile,
    mask: np.ndarray,
    label: str,
) -> None:
    mask_bool = mask.astype(bool)
    chains = data["chains"]
    chain_keep = data["mask"].astype(bool)
    residues = data["residues"]

    print(f"[mask-info] {record_id} {label}: atoms={int(mask_bool.sum())}/{mask_bool.shape[0]}")

    def _residue_entries(chain_idx: int) -> Iterable[str]:
        chain = chains[chain_idx]
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        for res in residues[res_start:res_end]:
            ra_start = int(res["atom_idx"])
            ra_end = ra_start + int(res["atom_num"])
            if mask_bool[ra_start:ra_end].any():
                yield f"{_to_str(res['name']).strip()}({int(res['res_idx'])})"

    for idx, chain in enumerate(chains):
        keep = idx < len(chain_keep) and chain_keep[idx]
        start = int(chain["atom_idx"])
        end = start + int(chain["atom_num"])
        if not mask_bool[start:end].any():
            continue
        entries = list(_residue_entries(idx))
        entries_str = ", ".join(entries) if entries else "None"
        print(
            f"  chain {_to_str(chain['name']).strip()} "
            f"(entity {int(chain['entity_id'])})"
            f"{'' if keep else ' [masked-out-chain]'}: {entries_str}"
        )


def _get_chain_name(chain: np.void) -> str:
    value = chain["name"] if "name" in chain.dtype.names else ""
    return _to_str(value).strip()


def _first_alpha(name: str) -> str:
    for ch in name.upper():
        if ch.isalpha():
            return ch
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject alignment/rmsd masks into structure NPZ files.")
    parser.add_argument(
        "structures_dir",
        type=Path,
        help="Directory containing processed structure NPZ files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report mask statistics without writing files.",
    )
    parser.add_argument(
        "--print-residues",
        action="store_true",
        help="When used with --dry-run, list the residues included in each mask.",
    )
    args = parser.parse_args()

    npz_paths = sorted(args.structures_dir.glob("*.npz"))
    if not npz_paths:
        raise SystemExit(f"No .npz files found under {args.structures_dir}")

    for npz_path in npz_paths:
        process_structure(
            npz_path,
            dry_run=args.dry_run,
            print_residues=args.print_residues,
        )

    if args.dry_run:
        print("Dry run complete; no files modified.")
    else:
        print(f"Processed {len(npz_paths)} structures.")


if __name__ == "__main__":
    main()
