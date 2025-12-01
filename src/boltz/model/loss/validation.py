import torch

from boltz.data import const
from boltz.model.loss.confidence import (
    compute_frame_pred,
    express_coordinate_in_frame,
    lddt_dist,
)
from boltz.model.loss.diffusion import weighted_rigid_align

from dataclasses import dataclass
from torch import Tensor


def factored_lddt_loss(
    true_atom_coords,
    pred_atom_coords,
    feats,
    atom_mask,
    multiplicity=1,
    cardinality_weighted=False,
):
    """Compute the lddt factorized into the different modalities.

    Parameters
    ----------
    true_atom_coords : torch.Tensor
        Ground truth atom coordinates after symmetry correction
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    feats : Dict[str, torch.Tensor]
        Input features
    atom_mask : torch.Tensor
        Atom mask
    multiplicity : int
        Diffusion batch size, by default 1

    Returns
    -------
    Dict[str, torch.Tensor]
        The lddt for each modality
    Dict[str, torch.Tensor]
        The total number of pairs for each modality

    """
    # extract necessary features
    atom_type = (
        torch.bmm(
            feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
        )
        .squeeze(-1)
        .long()
    )
    atom_type = atom_type.repeat_interleave(multiplicity, 0)

    ligand_mask = (atom_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (atom_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (atom_type == const.chain_type_ids["RNA"]).float()
    protein_mask = (atom_type == const.chain_type_ids["PROTEIN"]).float()

    nucleotide_mask = dna_mask + rna_mask

    true_d = torch.cdist(true_atom_coords, true_atom_coords)
    pred_d = torch.cdist(pred_atom_coords, pred_atom_coords)

    pair_mask = atom_mask[:, :, None] * atom_mask[:, None, :]
    pair_mask = (
        pair_mask
        * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
    )

    cutoff = 15 + 15 * (
        1 - (1 - nucleotide_mask[:, :, None]) * (1 - nucleotide_mask[:, None, :])
    )

    # compute different lddts
    dna_protein_mask = pair_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_lddt, dna_protein_total = lddt_dist(
        pred_d, true_d, dna_protein_mask, cutoff
    )
    del dna_protein_mask

    rna_protein_mask = pair_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_lddt, rna_protein_total = lddt_dist(
        pred_d, true_d, rna_protein_mask, cutoff
    )
    del rna_protein_mask

    ligand_protein_mask = pair_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_lddt, ligand_protein_total = lddt_dist(
        pred_d, true_d, ligand_protein_mask, cutoff
    )
    del ligand_protein_mask

    dna_ligand_mask = pair_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_lddt, dna_ligand_total = lddt_dist(
        pred_d, true_d, dna_ligand_mask, cutoff
    )
    del dna_ligand_mask

    rna_ligand_mask = pair_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_lddt, rna_ligand_total = lddt_dist(
        pred_d, true_d, rna_ligand_mask, cutoff
    )
    del rna_ligand_mask

    intra_dna_mask = pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_lddt, intra_dna_total = lddt_dist(pred_d, true_d, intra_dna_mask, cutoff)
    del intra_dna_mask

    intra_rna_mask = pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_lddt, intra_rna_total = lddt_dist(pred_d, true_d, intra_rna_mask, cutoff)
    del intra_rna_mask

    chain_id = feats["asym_id"]
    atom_chain_id = (
        torch.bmm(feats["atom_to_token"].float(), chain_id.unsqueeze(-1).float())
        .squeeze(-1)
        .long()
    )
    atom_chain_id = atom_chain_id.repeat_interleave(multiplicity, 0)
    same_chain_mask = (atom_chain_id[:, :, None] == atom_chain_id[:, None, :]).float()

    intra_ligand_mask = (
        pair_mask
        * same_chain_mask
        * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    )
    intra_ligand_lddt, intra_ligand_total = lddt_dist(
        pred_d, true_d, intra_ligand_mask, cutoff
    )
    del intra_ligand_mask

    intra_protein_mask = (
        pair_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_lddt, intra_protein_total = lddt_dist(
        pred_d, true_d, intra_protein_mask, cutoff
    )
    del intra_protein_mask

    protein_protein_mask = (
        pair_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_lddt, protein_protein_total = lddt_dist(
        pred_d, true_d, protein_protein_mask, cutoff
    )
    del protein_protein_mask

    lddt_dict = {
        "dna_protein": dna_protein_lddt,
        "rna_protein": rna_protein_lddt,
        "ligand_protein": ligand_protein_lddt,
        "dna_ligand": dna_ligand_lddt,
        "rna_ligand": rna_ligand_lddt,
        "intra_ligand": intra_ligand_lddt,
        "intra_dna": intra_dna_lddt,
        "intra_rna": intra_rna_lddt,
        "intra_protein": intra_protein_lddt,
        "protein_protein": protein_protein_lddt,
    }

    total_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }
    if not cardinality_weighted:
        for key in total_dict:
            total_dict[key] = (total_dict[key] > 0.0).float()

    return lddt_dict, total_dict


def factored_token_lddt_dist_loss(true_d, pred_d, feats, cardinality_weighted=False):
    """Compute the distogram lddt factorized into the different modalities.

    Parameters
    ----------
    true_d : torch.Tensor
        Ground truth atom distogram
    pred_d : torch.Tensor
        Predicted atom distogram
    feats : Dict[str, torch.Tensor]
        Input features

    Returns
    -------
    Tensor
        The lddt for each modality
    Tensor
        The total number of pairs for each modality

    """
    # extract necessary features
    token_type = feats["mol_type"]

    ligand_mask = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (token_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (token_type == const.chain_type_ids["RNA"]).float()
    protein_mask = (token_type == const.chain_type_ids["PROTEIN"]).float()
    nucleotide_mask = dna_mask + rna_mask

    token_mask = feats["token_disto_mask"]
    token_mask = token_mask[:, :, None] * token_mask[:, None, :]
    token_mask = token_mask * (1 - torch.eye(token_mask.shape[1])[None]).to(token_mask)

    cutoff = 15 + 15 * (
        1 - (1 - nucleotide_mask[:, :, None]) * (1 - nucleotide_mask[:, None, :])
    )

    # compute different lddts
    dna_protein_mask = token_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_lddt, dna_protein_total = lddt_dist(
        pred_d, true_d, dna_protein_mask, cutoff
    )

    rna_protein_mask = token_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_lddt, rna_protein_total = lddt_dist(
        pred_d, true_d, rna_protein_mask, cutoff
    )

    ligand_protein_mask = token_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_lddt, ligand_protein_total = lddt_dist(
        pred_d, true_d, ligand_protein_mask, cutoff
    )

    dna_ligand_mask = token_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_lddt, dna_ligand_total = lddt_dist(
        pred_d, true_d, dna_ligand_mask, cutoff
    )

    rna_ligand_mask = token_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_lddt, rna_ligand_total = lddt_dist(
        pred_d, true_d, rna_ligand_mask, cutoff
    )

    chain_id = feats["asym_id"]
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()
    intra_ligand_mask = (
        token_mask
        * same_chain_mask
        * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    )
    intra_ligand_lddt, intra_ligand_total = lddt_dist(
        pred_d, true_d, intra_ligand_mask, cutoff
    )

    intra_dna_mask = token_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_lddt, intra_dna_total = lddt_dist(pred_d, true_d, intra_dna_mask, cutoff)

    intra_rna_mask = token_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_lddt, intra_rna_total = lddt_dist(pred_d, true_d, intra_rna_mask, cutoff)

    chain_id = feats["asym_id"]
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()

    intra_protein_mask = (
        token_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_lddt, intra_protein_total = lddt_dist(
        pred_d, true_d, intra_protein_mask, cutoff
    )

    protein_protein_mask = (
        token_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_lddt, protein_protein_total = lddt_dist(
        pred_d, true_d, protein_protein_mask, cutoff
    )

    lddt_dict = {
        "dna_protein": dna_protein_lddt,
        "rna_protein": rna_protein_lddt,
        "ligand_protein": ligand_protein_lddt,
        "dna_ligand": dna_ligand_lddt,
        "rna_ligand": rna_ligand_lddt,
        "intra_ligand": intra_ligand_lddt,
        "intra_dna": intra_dna_lddt,
        "intra_rna": intra_rna_lddt,
        "intra_protein": intra_protein_lddt,
        "protein_protein": protein_protein_lddt,
    }

    total_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }

    if not cardinality_weighted:
        for key in total_dict:
            total_dict[key] = (total_dict[key] > 0.0).float()

    return lddt_dict, total_dict


def compute_plddt_mae(
    pred_atom_coords,
    feats,
    true_atom_coords,
    pred_lddt,
    true_coords_resolved_mask,
    multiplicity=1,
):
    """Compute the plddt mean absolute error.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    feats : torch.Tensor
        Input features
    true_atom_coords : torch.Tensor
        Ground truth atom coordinates
    pred_lddt : torch.Tensor
        Predicted lddt
    true_coords_resolved_mask : torch.Tensor
        Resolved atom mask
    multiplicity : int
        Diffusion batch size, by default 1

    Returns
    -------
    Tensor
        The mae for each modality
    Tensor
        The total number of pairs for each modality

    """
    # extract necessary features
    atom_mask = true_coords_resolved_mask
    R_set_to_rep_atom = feats["r_set_to_rep_atom"]
    R_set_to_rep_atom = R_set_to_rep_atom.repeat_interleave(multiplicity, 0).float()

    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)
    is_nucleotide_token = (token_type == const.chain_type_ids["DNA"]).float() + (
        token_type == const.chain_type_ids["RNA"]
    ).float()

    B = true_atom_coords.shape[0]

    atom_to_token = feats["atom_to_token"].float()
    atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

    token_to_rep_atom = feats["token_to_rep_atom"].float()
    token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)

    true_token_coords = torch.bmm(token_to_rep_atom, true_atom_coords)
    pred_token_coords = torch.bmm(token_to_rep_atom, pred_atom_coords)

    # compute true lddt
    true_d = torch.cdist(
        true_token_coords,
        torch.bmm(R_set_to_rep_atom, true_atom_coords),
    )
    pred_d = torch.cdist(
        pred_token_coords,
        torch.bmm(R_set_to_rep_atom, pred_atom_coords),
    )

    pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
    pair_mask = (
        pair_mask
        * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
    )
    pair_mask = torch.einsum("bnm,bkm->bnk", pair_mask, R_set_to_rep_atom)

    pair_mask = torch.bmm(token_to_rep_atom, pair_mask)
    atom_mask = torch.bmm(token_to_rep_atom, atom_mask.unsqueeze(-1).float()).squeeze(
        -1
    )
    is_nucleotide_R_element = torch.bmm(
        R_set_to_rep_atom, torch.bmm(atom_to_token, is_nucleotide_token.unsqueeze(-1))
    ).squeeze(-1)
    cutoff = 15 + 15 * is_nucleotide_R_element.reshape(B, 1, -1).repeat(
        1, true_d.shape[1], 1
    )

    target_lddt, mask_no_match = lddt_dist(
        pred_d, true_d, pair_mask, cutoff, per_atom=True
    )

    protein_mask = (
        (token_type == const.chain_type_ids["PROTEIN"]).float()
        * atom_mask
        * mask_no_match
    )
    ligand_mask = (
        (token_type == const.chain_type_ids["NONPOLYMER"]).float()
        * atom_mask
        * mask_no_match
    )
    dna_mask = (
        (token_type == const.chain_type_ids["DNA"]).float() * atom_mask * mask_no_match
    )
    rna_mask = (
        (token_type == const.chain_type_ids["RNA"]).float() * atom_mask * mask_no_match
    )

    protein_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * protein_mask) / (
        torch.sum(protein_mask) + 1e-5
    )
    protein_total = torch.sum(protein_mask)
    ligand_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * ligand_mask) / (
        torch.sum(ligand_mask) + 1e-5
    )
    ligand_total = torch.sum(ligand_mask)
    dna_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * dna_mask) / (
        torch.sum(dna_mask) + 1e-5
    )
    dna_total = torch.sum(dna_mask)
    rna_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * rna_mask) / (
        torch.sum(rna_mask) + 1e-5
    )
    rna_total = torch.sum(rna_mask)

    mae_plddt_dict = {
        "protein": protein_mae,
        "ligand": ligand_mae,
        "dna": dna_mae,
        "rna": rna_mae,
    }
    total_dict = {
        "protein": protein_total,
        "ligand": ligand_total,
        "dna": dna_total,
        "rna": rna_total,
    }

    return mae_plddt_dict, total_dict


def compute_pde_mae(
    pred_atom_coords,
    feats,
    true_atom_coords,
    pred_pde,
    true_coords_resolved_mask,
    multiplicity=1,
):
    """Compute the plddt mean absolute error.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    feats : torch.Tensor
        Input features
    true_atom_coords : torch.Tensor
        Ground truth atom coordinates
    pred_pde : torch.Tensor
        Predicted pde
    true_coords_resolved_mask : torch.Tensor
        Resolved atom mask
    multiplicity : int
        Diffusion batch size, by default 1

    Returns
    -------
    Tensor
        The mae for each modality
    Tensor
        The total number of pairs for each modality

    """
    # extract necessary features
    token_to_rep_atom = feats["token_to_rep_atom"].float()
    token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)

    token_mask = torch.bmm(
        token_to_rep_atom, true_coords_resolved_mask.unsqueeze(-1).float()
    ).squeeze(-1)

    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)

    true_token_coords = torch.bmm(token_to_rep_atom, true_atom_coords)
    pred_token_coords = torch.bmm(token_to_rep_atom, pred_atom_coords)

    # compute true pde
    true_d = torch.cdist(true_token_coords, true_token_coords)
    pred_d = torch.cdist(pred_token_coords, pred_token_coords)
    target_pde = (
        torch.clamp(
            torch.floor(torch.abs(true_d - pred_d) * 64 / 32).long(), max=63
        ).float()
        * 0.5
        + 0.25
    )

    pair_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)
    pair_mask = (
        pair_mask
        * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
    )

    protein_mask = (token_type == const.chain_type_ids["PROTEIN"]).float()
    ligand_mask = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (token_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (token_type == const.chain_type_ids["RNA"]).float()

    # compute different pdes
    dna_protein_mask = pair_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_mae = torch.sum(torch.abs(target_pde - pred_pde) * dna_protein_mask) / (
        torch.sum(dna_protein_mask) + 1e-5
    )
    dna_protein_total = torch.sum(dna_protein_mask)

    rna_protein_mask = pair_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_mae = torch.sum(torch.abs(target_pde - pred_pde) * rna_protein_mask) / (
        torch.sum(rna_protein_mask) + 1e-5
    )
    rna_protein_total = torch.sum(rna_protein_mask)

    ligand_protein_mask = pair_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_mae = torch.sum(
        torch.abs(target_pde - pred_pde) * ligand_protein_mask
    ) / (torch.sum(ligand_protein_mask) + 1e-5)
    ligand_protein_total = torch.sum(ligand_protein_mask)

    dna_ligand_mask = pair_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_mae = torch.sum(torch.abs(target_pde - pred_pde) * dna_ligand_mask) / (
        torch.sum(dna_ligand_mask) + 1e-5
    )
    dna_ligand_total = torch.sum(dna_ligand_mask)

    rna_ligand_mask = pair_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_mae = torch.sum(torch.abs(target_pde - pred_pde) * rna_ligand_mask) / (
        torch.sum(rna_ligand_mask) + 1e-5
    )
    rna_ligand_total = torch.sum(rna_ligand_mask)

    intra_ligand_mask = pair_mask * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    intra_ligand_mae = torch.sum(
        torch.abs(target_pde - pred_pde) * intra_ligand_mask
    ) / (torch.sum(intra_ligand_mask) + 1e-5)
    intra_ligand_total = torch.sum(intra_ligand_mask)

    intra_dna_mask = pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_mae = torch.sum(torch.abs(target_pde - pred_pde) * intra_dna_mask) / (
        torch.sum(intra_dna_mask) + 1e-5
    )
    intra_dna_total = torch.sum(intra_dna_mask)

    intra_rna_mask = pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_mae = torch.sum(torch.abs(target_pde - pred_pde) * intra_rna_mask) / (
        torch.sum(intra_rna_mask) + 1e-5
    )
    intra_rna_total = torch.sum(intra_rna_mask)

    chain_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()

    intra_protein_mask = (
        pair_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_mae = torch.sum(
        torch.abs(target_pde - pred_pde) * intra_protein_mask
    ) / (torch.sum(intra_protein_mask) + 1e-5)
    intra_protein_total = torch.sum(intra_protein_mask)

    protein_protein_mask = (
        pair_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_mae = torch.sum(
        torch.abs(target_pde - pred_pde) * protein_protein_mask
    ) / (torch.sum(protein_protein_mask) + 1e-5)
    protein_protein_total = torch.sum(protein_protein_mask)

    mae_pde_dict = {
        "dna_protein": dna_protein_mae,
        "rna_protein": rna_protein_mae,
        "ligand_protein": ligand_protein_mae,
        "dna_ligand": dna_ligand_mae,
        "rna_ligand": rna_ligand_mae,
        "intra_ligand": intra_ligand_mae,
        "intra_dna": intra_dna_mae,
        "intra_rna": intra_rna_mae,
        "intra_protein": intra_protein_mae,
        "protein_protein": protein_protein_mae,
    }
    total_pde_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }

    return mae_pde_dict, total_pde_dict


def compute_pae_mae(
    pred_atom_coords,
    feats,
    true_atom_coords,
    pred_pae,
    true_coords_resolved_mask,
    multiplicity=1,
):
    """Compute the pae mean absolute error.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    feats : torch.Tensor
        Input features
    true_atom_coords : torch.Tensor
        Ground truth atom coordinates
    pred_pae : torch.Tensor
        Predicted pae
    true_coords_resolved_mask : torch.Tensor
        Resolved atom mask
    multiplicity : int
        Diffusion batch size, by default 1

    Returns
    -------
    Tensor
        The mae for each modality
    Tensor
        The total number of pairs for each modality

    """
    # Retrieve frames and resolved masks
    frames_idx_original = feats["frames_idx"]
    mask_frame_true = feats["frame_resolved_mask"]

    # Adjust the frames for nonpolymers after symmetry correction!
    # NOTE: frames of polymers do not change under symmetry!
    frames_idx_true, mask_collinear_true = compute_frame_pred(
        true_atom_coords,
        frames_idx_original,
        feats,
        multiplicity,
        resolved_mask=true_coords_resolved_mask,
    )

    frame_true_atom_a, frame_true_atom_b, frame_true_atom_c = (
        frames_idx_true[:, :, :, 0],
        frames_idx_true[:, :, :, 1],
        frames_idx_true[:, :, :, 2],
    )
    # Compute token coords in true frames
    B, N, _ = true_atom_coords.shape
    true_atom_coords = true_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    true_coords_transformed = express_coordinate_in_frame(
        true_atom_coords, frame_true_atom_a, frame_true_atom_b, frame_true_atom_c
    )

    # Compute pred frames and mask
    frames_idx_pred, mask_collinear_pred = compute_frame_pred(
        pred_atom_coords, frames_idx_original, feats, multiplicity
    )
    frame_pred_atom_a, frame_pred_atom_b, frame_pred_atom_c = (
        frames_idx_pred[:, :, :, 0],
        frames_idx_pred[:, :, :, 1],
        frames_idx_pred[:, :, :, 2],
    )
    # Compute token coords in pred frames
    B, N, _ = pred_atom_coords.shape
    pred_atom_coords = pred_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    pred_coords_transformed = express_coordinate_in_frame(
        pred_atom_coords, frame_pred_atom_a, frame_pred_atom_b, frame_pred_atom_c
    )

    target_pae_continuous = torch.sqrt(
        ((true_coords_transformed - pred_coords_transformed) ** 2).sum(-1) + 1e-8
    )
    target_pae = (
        torch.clamp(torch.floor(target_pae_continuous * 64 / 32).long(), max=63).float()
        * 0.5
        + 0.25
    )

    # Compute mask for the pae loss
    b_true_resolved_mask = true_coords_resolved_mask[
        torch.arange(B // multiplicity)[:, None, None].to(
            pred_coords_transformed.device
        ),
        frame_true_atom_b,
    ]

    pair_mask = (
        mask_frame_true[:, None, :, None]  # if true frame is invalid
        * mask_collinear_true[:, :, :, None]  # if true frame is invalid
        * mask_collinear_pred[:, :, :, None]  # if pred frame is invalid
        * b_true_resolved_mask[:, :, None, :]  # If atom j is not resolved
        * feats["token_pad_mask"][:, None, :, None]
        * feats["token_pad_mask"][:, None, None, :]
    )

    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)

    protein_mask = (token_type == const.chain_type_ids["PROTEIN"]).float()
    ligand_mask = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (token_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (token_type == const.chain_type_ids["RNA"]).float()

    # compute different paes
    dna_protein_mask = pair_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_mae = torch.sum(torch.abs(target_pae - pred_pae) * dna_protein_mask) / (
        torch.sum(dna_protein_mask) + 1e-5
    )
    dna_protein_total = torch.sum(dna_protein_mask)

    rna_protein_mask = pair_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_mae = torch.sum(torch.abs(target_pae - pred_pae) * rna_protein_mask) / (
        torch.sum(rna_protein_mask) + 1e-5
    )
    rna_protein_total = torch.sum(rna_protein_mask)

    ligand_protein_mask = pair_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * ligand_protein_mask
    ) / (torch.sum(ligand_protein_mask) + 1e-5)
    ligand_protein_total = torch.sum(ligand_protein_mask)

    dna_ligand_mask = pair_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_mae = torch.sum(torch.abs(target_pae - pred_pae) * dna_ligand_mask) / (
        torch.sum(dna_ligand_mask) + 1e-5
    )
    dna_ligand_total = torch.sum(dna_ligand_mask)

    rna_ligand_mask = pair_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_mae = torch.sum(torch.abs(target_pae - pred_pae) * rna_ligand_mask) / (
        torch.sum(rna_ligand_mask) + 1e-5
    )
    rna_ligand_total = torch.sum(rna_ligand_mask)

    intra_ligand_mask = pair_mask * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    intra_ligand_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * intra_ligand_mask
    ) / (torch.sum(intra_ligand_mask) + 1e-5)
    intra_ligand_total = torch.sum(intra_ligand_mask)

    intra_dna_mask = pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_mae = torch.sum(torch.abs(target_pae - pred_pae) * intra_dna_mask) / (
        torch.sum(intra_dna_mask) + 1e-5
    )
    intra_dna_total = torch.sum(intra_dna_mask)

    intra_rna_mask = pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_mae = torch.sum(torch.abs(target_pae - pred_pae) * intra_rna_mask) / (
        torch.sum(intra_rna_mask) + 1e-5
    )
    intra_rna_total = torch.sum(intra_rna_mask)

    chain_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()

    intra_protein_mask = (
        pair_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * intra_protein_mask
    ) / (torch.sum(intra_protein_mask) + 1e-5)
    intra_protein_total = torch.sum(intra_protein_mask)

    protein_protein_mask = (
        pair_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * protein_protein_mask
    ) / (torch.sum(protein_protein_mask) + 1e-5)
    protein_protein_total = torch.sum(protein_protein_mask)

    mae_pae_dict = {
        "dna_protein": dna_protein_mae,
        "rna_protein": rna_protein_mae,
        "ligand_protein": ligand_protein_mae,
        "dna_ligand": dna_ligand_mae,
        "rna_ligand": rna_ligand_mae,
        "intra_ligand": intra_ligand_mae,
        "intra_dna": intra_dna_mae,
        "intra_rna": intra_rna_mae,
        "intra_protein": intra_protein_mae,
        "protein_protein": protein_protein_mae,
    }
    total_pae_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }

    return mae_pae_dict, total_pae_dict


def weighted_minimum_rmsd(
    pred_atom_coords,
    feats,
    multiplicity=1,
    nucleotide_weight=5.0,
    ligand_weight=10.0,
    alignment_mask=None,
    rmsd_mask=None,
    ca_only: bool = False,
):
    """Compute rmsd of the aligned atom coordinates.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    feats : torch.Tensor
        Input features
    multiplicity : int
        Diffusion batch size, by default 1

    Returns
    -------
    Tensor
        The rmsds
    Tensor
        The best rmsd

    """
    atom_coords = feats["coords"]
    atom_coords = atom_coords.repeat_interleave(multiplicity, 0)
    atom_coords = atom_coords[:, 0]

    atom_mask = feats["atom_resolved_mask"]
    atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
    print(f'##### inside validation weighted_minimum_rmsd: atom_mask.shape = {atom_mask.shape} #####')

    target_len = pred_atom_coords.shape[0]
    base_len = feats["coords"].shape[0]

    def _prepare_mask(mask, fallback):
        if mask is None:
            prepared = fallback
        else:
            if mask.shape[0] == target_len:
                prepared = mask
            elif mask.shape[0] == base_len and base_len * multiplicity == target_len:
                prepared = mask.repeat_interleave(multiplicity, 0)
            else:
                raise ValueError(
                    "Unexpected mask shape: "
                    f"{mask.shape[0]} (expected {target_len} or {base_len})."
                )
        return prepared.to(dtype=fallback.dtype, device=pred_atom_coords.device)

    align_mask = _prepare_mask(alignment_mask, atom_mask)
    # print(f'##### inside validation weighted_minimum_rmsd: rmsd shape: {rmsd_mask.shape if rmsd_mask is not None else 0} and rmsdmask.sum = {rmsd_mask.sum() if rmsd_mask is not None else 0} #####')
    calc_mask = _prepare_mask(rmsd_mask, atom_mask)

    align_weights = atom_coords.new_ones(atom_coords.shape[:2])
    atom_type = (
        torch.bmm(
            feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
        )
        .squeeze(-1)
        .long()
    )
    atom_type = atom_type.repeat_interleave(multiplicity, 0)

    align_weights = align_weights * (
        1
        + nucleotide_weight
        * (
            torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
            + torch.eq(atom_type, const.chain_type_ids["RNA"]).float()
        )
        + ligand_weight
        * torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()
    )

    # CA-only filtering (apply to both alignment and calc masks)
    if ca_only and "asym_id" in feats and "residue_index" in feats:
        asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
        residue_index = feats["residue_index"].repeat_interleave(multiplicity, 0)
        atom_to_tok = feats["atom_to_token"].repeat_interleave(multiplicity, 0)
        ca_mask = torch.zeros_like(calc_mask, dtype=torch.bool, device=calc_mask.device)
        combined = (align_mask | calc_mask).bool()
        for b in range(pred_atom_coords.shape[0]):
            tok_idx = atom_to_tok[b].argmax(dim=-1)
            chain_ids = asym_id[b][tok_idx]
            res_ids = residue_index[b][tok_idx]
            seen = set()
            for idx in torch.nonzero(combined[b], as_tuple=False).squeeze(-1):
                key = (int(chain_ids[idx].item()), int(res_ids[idx].item()))
                if key in seen:
                    continue
                seen.add(key)
                ca_mask[b, idx] = True
        align_mask = align_mask & ca_mask
        calc_mask = calc_mask & ca_mask

    with torch.no_grad():
        atom_coords_aligned_ground_truth = weighted_rigid_align(
            atom_coords, pred_atom_coords, align_weights, mask=align_mask
        )

    # weighted MSE loss of denoised atom positions
    mse_loss = ((pred_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(dim=-1)
    # print(f'##### inside validation weighted_minimum_rmsd: mse_loss.sum = {mse_loss.sum()} #####')
    # print(f'##### inside validation weighted_minimum_rmsd: calc_mask.sum = {calc_mask.sum()} #####')
    denom = torch.sum(align_weights * calc_mask, dim=-1)
    rmsd = torch.sqrt(
        torch.sum(mse_loss * align_weights * calc_mask, dim=-1)
        / torch.clamp(denom, min=1e-8)
    )
    best_rmsd = torch.min(rmsd.reshape(-1, multiplicity), dim=1).values

    print(f'##### rmsd and best rmsd values: {rmsd}, {best_rmsd} #####')

    return rmsd, best_rmsd

def weighted_minimum_rmsd_single(
    pred_atom_coords,
    atom_coords,
    atom_mask,
    atom_to_token,
    mol_type,
    nucleotide_weight=5.0,
    ligand_weight=10.0,
    alignment_mask=None,
    rmsd_mask=None,
    asym_id=None,
    residue_index=None,
    chain_names=None,
    debug_alignment: bool = False,
    debug_alignment_samples: int = 6,
    ca_only: bool = True,
    debug_distances: bool = False,
    skip_alignment: bool = False,
):
    """Compute rmsd of the aligned atom coordinates.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    atom_coords: torch.Tensor
        Ground truth atom coordinates
    atom_mask : torch.Tensor
        Resolved atom mask
    atom_to_token : torch.Tensor
        Atom to token mapping
    mol_type : torch.Tensor
        Atom type

    Returns
    -------
    Tensor
        The rmsd
    Tensor
        The aligned coordinates
    Tensor
        The aligned weights

    """
    align_weights = atom_coords.new_ones(atom_coords.shape[:2])
    atom_type = (
        torch.bmm(atom_to_token.float(), mol_type.unsqueeze(-1).float())
        .squeeze(-1)
        .long()
    )

    align_weights = align_weights * (
        1
        + nucleotide_weight
        * (
            torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
            + torch.eq(atom_type, const.chain_type_ids["RNA"]).float()
        )
        + ligand_weight
        * torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()
    )

    with torch.no_grad():
        if skip_alignment:
            atom_coords_aligned_ground_truth = atom_coords
        else:
            mask_align = alignment_mask if alignment_mask is not None else atom_mask
            mask_align = mask_align.bool() & atom_mask.bool()
            atom_coords_aligned_ground_truth = weighted_rigid_align(
                atom_coords, pred_atom_coords, align_weights, mask=mask_align
            )

    # weighted MSE loss of denoised atom positions
    calc_mask = rmsd_mask if rmsd_mask is not None else atom_mask
    calc_mask = calc_mask.bool() & atom_mask.bool()

    if ca_only and asym_id is not None and residue_index is not None:
        # Keep only one atom (CA) per residue in the calc_mask
        ca_mask = torch.zeros_like(calc_mask, dtype=torch.bool)
        batch_size = calc_mask.shape[0]
        for b in range(batch_size):
            atom_to_tok_idx = atom_to_token[b].argmax(dim=-1)
            chain_ids = asym_id[b][atom_to_tok_idx]
            res_ids = residue_index[b][atom_to_tok_idx]
            seen = set()
            for idx in torch.nonzero(calc_mask[b], as_tuple=False).squeeze(-1):
                rid = (int(chain_ids[idx]), int(res_ids[idx]))
                if rid in seen:
                    continue
                seen.add(rid)
                ca_mask[b, idx] = True
        calc_mask = calc_mask & ca_mask

    mse_loss = ((pred_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(dim=-1)
    denom = torch.sum(align_weights * calc_mask, dim=-1)
    rmsd = torch.sqrt(
        torch.sum(mse_loss * align_weights * calc_mask, dim=-1)
        / torch.clamp(denom, min=1e-8)
    )

    if debug_alignment and asym_id is not None and residue_index is not None:
        batch_size = mse_loss.shape[0]
        for b in range(batch_size):
            atom_to_tok_idx = atom_to_token[b].argmax(dim=-1)
            chain_ids = asym_id[b][atom_to_tok_idx]
            res_ids = residue_index[b][atom_to_tok_idx]
            chain_types = (
                torch.bmm(
                    atom_to_token[b].unsqueeze(0).float(),
                    mol_type[b].unsqueeze(0).unsqueeze(-1).float(),
                )
                .squeeze(0)
                .squeeze(-1)
                .long()
            )
            calc_mask_b = calc_mask[b] > 0
            # When skip_alignment=True the reference is assumed pre-aligned, so show
            # the predicted coords as the "after" position to keep the debug useful.
            base_coords = atom_coords[b]
            aligned_coords = (
                pred_atom_coords[b]
                if skip_alignment
                else atom_coords_aligned_ground_truth[b]
            )

            for chain in torch.unique(chain_ids):
                chain_token_mask = chain_ids == chain
                protein_mask = chain_types != const.chain_type_ids["NONPOLYMER"]
                if not (chain_token_mask & protein_mask).any():
                    continue

                chain_res_ids = torch.unique(res_ids[chain_token_mask])
                if chain_res_ids.numel() == 0:
                    continue
                per_res_rmsd = []
                for rid in chain_res_ids:
                    # CA-only: take the first atom for each residue (assumed CA after masking)
                    res_mask = (
                        calc_mask_b & chain_token_mask & (res_ids == rid) & protein_mask
                    )
                    if res_mask.any():
                        atom_indices = torch.nonzero(res_mask, as_tuple=False).squeeze(-1)
                        if atom_indices.numel() == 0:
                            continue
                        ca_idx = atom_indices[0]
                        ca_mask = torch.zeros_like(res_mask)
                        ca_mask[ca_idx] = True
                        rms_val = torch.sqrt(
                            torch.clamp(mse_loss[b][ca_mask].mean(), min=0.0)
                        ).item()
                        per_res_rmsd.append((int(rid.item()), rms_val, int(ca_idx.item())))
                if not per_res_rmsd:
                    continue
                per_res_rmsd.sort(key=lambda x: x[1], reverse=True)
                take = per_res_rmsd[:debug_alignment_samples]

                chain_label = None
                if chain_names is not None:
                    try:
                        chain_label = chain_names[b][int(chain.item())]
                    except Exception:
                        pass
                chain_display = (
                    f"{chain_label} (id={int(chain.item())})"
                    if chain_label not in (None, "None")
                    else f"id={int(chain.item())}"
                )

                # print(f"[alignment-debug] chain {chain_display}")
                # for rid, rms_val, ca_idx in take:
                #     atom_mask_res = (
                #         calc_mask_b & chain_token_mask & (res_ids == rid) & protein_mask
                #     )
                #     if not atom_mask_res.any():
                #         continue
                #     before = base_coords[ca_idx : ca_idx + 1].detach().cpu().numpy().tolist()
                #     after = aligned_coords[ca_idx : ca_idx + 1].detach().cpu().numpy().tolist()
                #     print(
                #         f"  residue {rid}: rmsd={rms_val:.3f}, atom_idx={ca_idx}, CA before {before} after {after}"
                #     )

    return rmsd, atom_coords_aligned_ground_truth, align_weights

@dataclass
class SampleMetrics:
    sample_idx: int
    rmsd_whole: float
    rmsd_masked: float
