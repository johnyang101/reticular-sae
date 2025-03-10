import os
import argparse
from typing import Tuple
from eval.utils import parse_pdb_feats, align_feats, calc_aligned_rmsd

def calculate_rmsd_metrics(
    pdb_name: str,
    base_dir: str,
    exp_subdir: str = "exp_pdbs",
    esmf_no_abl_subdir: str = "esmf_no_abl_preds",
    esmf_abl_full_subdir: str = "esmf_abl_full_preds",
    esmf_abl_keep_l36_subdir: str = "esmf_abl_keep_l36_preds",
    sae_subdir: str = "sae_preds",
    exp_chain_id: str = " ", #NOTE: ' ' is the chain id for the PDB files we generated.
    esmf_chain_id: str = "A", #NOTE: We only evaluate on single chain proteins so we hardcode chain id to 'A'
    sae_chain_id: str = "A" #NOTE: We only evaluate on single chain proteins so we hardcode chain id to 'A'
) -> Tuple[float, float, float, float]:
    """
    Calculate RMSD metrics between experimental structure and various model predictions.
    
    Args:
        pdb_name: Name of the PDB file (without extension)
        base_dir: Base directory containing all PDB files
        exp_subdir: Subdirectory for experimental PDB files
        esmf_no_abl_subdir: Subdirectory for ESMFold no ablation predictions
        esmf_abl_full_subdir: Subdirectory for ESMFold full ablation predictions
        esmf_abl_keep_l36_subdir: Subdirectory for ESMFold keep l36 ablation predictions
        sae_subdir: Subdirectory for SAE predictions
        exp_chain_id: Chain id for experimental structure
        esmf_chain_id: Chain id for ESMFold predictions
        sae_chain_id: Chain id for SAE predictions

    Returns:
        Tuple of RMSD values:
        - exp vs ESMFold no ablation
        - exp vs ESMFold ablation full
        - exp vs ESMFold ablation keep l36
        - SAE vs ESMFold ablation keep l36
    """
    # Construct paths
    paths = {
        'exp': os.path.join(base_dir, exp_subdir, f'{pdb_name}.pdb'),
        'esmf_no_abl': os.path.join(base_dir, esmf_no_abl_subdir, f'{pdb_name}.pdb'),
        'esmf_abl_full': os.path.join(base_dir, esmf_abl_full_subdir, f'{pdb_name}.pdb'),
        'esmf_abl_keep_l36': os.path.join(base_dir, esmf_abl_keep_l36_subdir, f'{pdb_name}.pdb'),
        'sae': os.path.join(base_dir, sae_subdir, f'{pdb_name}.pdb')
    }
    
    # Verify all files exist
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find {name} PDB file at: {path}")
    
    try:
        # Load experimental structure features
        exp_feats = parse_pdb_feats(pdb_name, paths['exp'])[exp_chain_id]
        
        # Calculate RMSDs for each model
        results = {}
        
        # ESMFold no ablation
        esmf_no_abl_feats = parse_pdb_feats(pdb_name, paths['esmf_no_abl'])[esmf_chain_id]
        pos1, pos2 = align_feats(exp_feats, esmf_no_abl_feats)
        results['exp_vs_esmf_no_abl'] = calc_aligned_rmsd(pos1, pos2)
        
        # ESMFold ablation full
        esmf_abl_full_feats = parse_pdb_feats(pdb_name, paths['esmf_abl_full'])[esmf_chain_id]
        pos1, pos2 = align_feats(exp_feats, esmf_abl_full_feats)
        results['exp_vs_esmf_abl_full'] = calc_aligned_rmsd(pos1, pos2)
        
        # ESMFold ablation keep l36
        esmf_abl_keep_l36_feats = parse_pdb_feats(pdb_name, paths['esmf_abl_keep_l36'])[esmf_chain_id]
        pos1, pos2 = align_feats(exp_feats, esmf_abl_keep_l36_feats)
        results['exp_vs_esmf_abl_keep_l36'] = calc_aligned_rmsd(pos1, pos2)
        
        # SAE vs ESMFold ablation keep l36
        sae_feats = parse_pdb_feats(pdb_name, paths['sae'])[sae_chain_id]
        pos1, pos2 = align_feats(esmf_abl_keep_l36_feats, sae_feats)
        results['sae_vs_esmf_abl_keep_l36'] = calc_aligned_rmsd(pos1, pos2)
        
        # Print results
        print(f"\nResults for PDB: {pdb_name}")
        print("-" * 50)
        for name, rmsd in results.items():
            print(f"{name}: {rmsd:.4f}")
        
        return tuple(results.values())
    
    except Exception as e:
        print(f"Error processing {pdb_name}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Calculate RMSD metrics for protein structure predictions')
    parser.add_argument('pdb_name', help='Name of the PDB file (without extension)')
    parser.add_argument('--base_dir', required=True, help='Base directory containing all PDB files')
    parser.add_argument('--exp_subdir', default='exp_pdbs', help='Subdirectory for experimental PDBs')
    parser.add_argument('--esmf_no_abl_subdir', default='esmf_no_abl_preds', help='Subdirectory for ESMFold no ablation predictions')
    parser.add_argument('--esmf_abl_full_subdir', default='esmf_abl_full_preds', help='Subdirectory for ESMFold full ablation predictions')
    parser.add_argument('--esmf_abl_keep_l36_subdir', default='esmf_abl_keep_l36_preds', help='Subdirectory for ESMFold keep l36 ablation predictions')
    parser.add_argument('--sae_subdir', default='sae_preds', help='Subdirectory for SAE predictions')
    parser.add_argument('--exp_chain_id', default=' ', help='Chain id for experimental structure')
    parser.add_argument('--esmf_chain_id', default='A', help='Chain id for ESMFold predictions')
    parser.add_argument('--sae_chain_id', default='A', help='Chain id for SAE predictions')
    args = parser.parse_args()
    
    calculate_rmsd_metrics(
        args.pdb_name,
        args.base_dir,
        args.exp_subdir,
        args.esmf_no_abl_subdir,
        args.esmf_abl_full_subdir,
        args.esmf_abl_keep_l36_subdir,
        args.sae_subdir,
        args.exp_chain_id,
        args.esmf_chain_id,
        args.sae_chain_id
    )

if __name__ == "__main__":
    main()