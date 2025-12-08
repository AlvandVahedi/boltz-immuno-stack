from __future__ import annotations

from typing import Tuple

import numpy as np

from boltz.data.types import Record, Structure

ALIGNMENT_PREFIXES = ("A", "B")
RMSD_PREFIXES = ("C", "D")


def ensure_cyclic_period_field(chains: np.ndarray) -> np.ndarray:
    if "cyclic_period" in chains.dtype.names:
        return chains

    dtype = chains.dtype.descr + [("cyclic_period", "<i4")]
    new_chains = np.empty(chains.shape, dtype=dtype)
    for name in chains.dtype.names:
        new_chains[name] = chains[name]
    new_chains["cyclic_period"] = 0
    return new_chains


def build_chain_masks(structure: Structure, record: Record) -> Tuple[np.ndarray, np.ndarray]:
    """Build alignment and RMSD masks per atom using manifest metadata."""

    num_atoms = len(structure.atoms)
    alignment_mask = np.zeros(num_atoms, dtype=bool)
    rmsd_mask = np.zeros(num_atoms, dtype=bool)

    if record is None or not getattr(record, "chains", None):
        return alignment_mask, rmsd_mask

    asym_id_to_chain = {int(chain["asym_id"]): chain for chain in structure.chains}

    for chain in record.chains:
        if not getattr(chain, "valid", True):
            continue

        chain_array = asym_id_to_chain.get(int(chain.chain_id))
        if chain_array is None:
            continue

        chain_name = (chain.chain_name or "").upper()
        start = int(chain_array["atom_idx"])
        end = start + int(chain_array["atom_num"])

        if chain_name.startswith(ALIGNMENT_PREFIXES):
            alignment_mask[start:end] = True
        if chain_name.startswith(RMSD_PREFIXES):
            rmsd_mask[start:end] = True

    return alignment_mask, rmsd_mask
