"""Utility functions for protein structure analysis and manipulation."""

# Standard library imports
import dataclasses
from typing import Dict, List, Tuple, Union

# Third party imports
import numpy as np
from Bio import PDB, pairwise2
from Bio.PDB import Chain

# Local imports
import eval.residue_constants as residue_constants
from eval.protein import Protein

# Constants
CHAIN_FEATS = [
    'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'b_factors'
]

# Feature parsing and processing functions
def parse_chain_feats(chain_feats: Dict, scale_factor: float = 1.) -> Dict:
    """Parse and process chain features.
    
    Args:
        chain_feats: Dictionary containing chain features
        scale_factor: Factor to scale atom positions
        
    Returns:
        Processed chain features dictionary
    """
    ca_idx = residue_constants.atom_order['CA']
    chain_feats['bb_mask'] = chain_feats['atom_mask'][:, ca_idx]
    bb_pos = chain_feats['atom_positions'][:, ca_idx]
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['bb_mask']) + 1e-5)
    centered_pos = chain_feats['atom_positions'] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    chain_feats['atom_positions'] = scaled_pos * chain_feats['atom_mask'][..., None]
    chain_feats['bb_positions'] = chain_feats['atom_positions'][:, ca_idx]
    return chain_feats

def process_chain(chain: Chain, chain_id: str) -> Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.

    Forked from alphafold.common.protein.from_pdb_string

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.

    Took out lines 110-112 since that would mess up CDR numbering.

    Args:
        chain: Instance of Biopython's chain class.
        chain_id: ID of the chain

    Returns:
        Protein object with protein features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num)
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.
            res_b_factors[residue_constants.atom_order[atom.name]
                          ] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors))

def parse_pdb_feats(
        pdb_name: str,
        pdb_path: str,
        scale_factor: float = 1.,
    ) -> Union[Dict, Dict[str, Dict]]:
    """Parse features from a PDB file.
    
    Args:
        pdb_name: name of PDB to parse.
        pdb_path: path to PDB file to read.
        scale_factor: factor to scale atom positions.
        
    Returns:
        Dict with CHAIN_FEATS features extracted from PDB with specified
        preprocessing.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_path)
    struct_chains = {
        chain.id: chain
        for chain in structure.get_chains()}
    # print(struct_chains)

    def _process_chain_id(x):
        chain_prot = process_chain(struct_chains[x], x)
        chain_dict = dataclasses.asdict(chain_prot)

        # Process features
        feat_dict = {x: chain_dict[x] for x in CHAIN_FEATS}
        return parse_chain_feats(
            feat_dict, scale_factor=scale_factor)

    chain_id = list(struct_chains.keys())

    if isinstance(chain_id, str):
        return _process_chain_id(chain_id)
    elif isinstance(chain_id, list):
        return {
            x: _process_chain_id(x) for x in chain_id
        }
    elif chain_id is None:
        return {
            x: _process_chain_id(x) for x in struct_chains
        }
    else:
        raise ValueError(f'Unrecognized chain list {chain_id}')

# Structural alignment and transformation functions
def rigid_transform_3D(A: np.ndarray, B: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Calculate the optimal rigid transform that aligns A to B.
    
    Args:
        A: (N,3) array of points to transform
        B: (N,3) array of target points
        verbose: Whether to print additional information
        
    Returns:
        Tuple containing:
        - transformed A coordinates
        - rotation matrix R
        - translation vector t
        - whether reflection was detected
    """
    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    reflection_detected = False
    if np.linalg.det(R) < 0:
        if verbose:
            print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
        reflection_detected = True

    t = -R @ centroid_A + centroid_B
    optimal_A = R @ A + t

    return optimal_A.T, R, t, reflection_detected

def align_feats(feats_1: Dict, feats_2: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Aligns two feature dictionaries based on sequence alignment.
    
    Args:
        feats_1: First feature dict containing 'aatype' and 'bb_positions'
        feats_2: Second feature dict containing 'aatype' and 'bb_positions'
        
    Returns:
        Tuple of aligned backbone positions (pos_1, pos_2)
    """
    aatype_to_seq = lambda aatype: ''.join(residue_constants.restypes_with_x[x] for x in aatype)
    # Convert aatype arrays to sequences
    seq_1 = aatype_to_seq(feats_1['aatype'])
    seq_2 = aatype_to_seq(feats_2['aatype'])
    
    # Get sequence alignment
    alignment = pairwise2.align.globalxx(seq_1, seq_2)[0]
    aligned_1, aligned_2 = alignment[0], alignment[1]
    
    # Find matching positions (excluding gaps)
    match_indices = []
    idx1 = 0
    idx2 = 0
    for a, b in zip(aligned_1, aligned_2):
        if a != '-' and b != '-':
            match_indices.append((idx1, idx2))
            idx1 += 1
            idx2 += 1
        elif a == '-':
            idx2 += 1
        else:  # b == '-'
            idx1 += 1
    
    if not match_indices:
        raise ValueError("No matching residues found in alignment")
        
    # Extract indices for each sequence
    idx_1, idx_2 = zip(*match_indices)
    
    # Get aligned positions
    pos_1 = feats_1['bb_positions'][list(idx_1)]
    pos_2 = feats_2['bb_positions'][list(idx_2)]
    
    return pos_1, pos_2

def calc_aligned_rmsd(pos_1, pos_2):
    aligned_pos_1 = rigid_transform_3D(pos_1, pos_2)[0]
    return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1))