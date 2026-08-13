"""Split a multi-molecule SDF into one 3D-prepared file per DrugBank ID.

Reads a combined SDF (e.g. DrugBank's approved-structures export), and for each
molecule adds hydrogens, embeds 3D coordinates, optimises the geometry and writes
``<DRUGBANK_ID>.sdf`` into the output folder. That folder is what the pipeline
passes as ``--ligands``: every docking job looks up its ligand by DrugBank ID, so
a molecule with no ``DRUGBANK_ID`` property is skipped and reported rather than
written under a guessed name.

Run once per ligand set, ahead of the pipeline.

Usage::

    python3 split_sdf.py <input.sdf> <output_dir>
"""

from rdkit import Chem
from rdkit.Chem import AllChem # Contains functions for 3D operations
import os
import sys
from tqdm import tqdm

def split_and_process_sdf(input_sdf_file_path, output_directory_path, remove_hydrogens_before_save=False):
    """
    Loads an SDF file, processes each molecule to add hydrogens, build 3D coordinates,
    optimize geometry, and then saves each to a separate SDF file named by its <DRUGBANK_ID>.

    Args:
        input_sdf_file_path (str): Full path to the input SDF file.
        output_directory_path (str): Full path to the directory where individual
                                     SDF files will be saved. Will be created
                                     if it doesn't exist.
        remove_hydrogens_before_save (bool): If True, hydrogens added for 3D processing
                                             will be removed before saving the SDF file.
                                             Set to False to keep them (recommended for 3D data).
    """
    if not os.path.exists(input_sdf_file_path):
        print(f"Error: Input SDF file not found at '{input_sdf_file_path}'")
        sys.exit(1)

    if not os.path.isdir(output_directory_path):
        try:
            os.makedirs(output_directory_path)
            print(f"Created output directory: '{output_directory_path}'")
        except OSError as e:
            print(f"Error: Could not create output directory '{output_directory_path}': {e}")
            sys.exit(1)

    print(f"Processing input file: '{input_sdf_file_path}'")
    print(f"Saving processed molecules to: '{output_directory_path}'")
    print(f"Remove hydrogens before saving: {remove_hydrogens_before_save}")

    supplier = Chem.SDMolSupplier(input_sdf_file_path)
    
    molecules_processed = 0
    molecules_saved = 0
    molecules_failed_3d = 0
    molecules_failed_opt = 0

    for i, mol in enumerate(tqdm(supplier, desc="Processing Molecules", unit="mol")):
        molecules_processed += 1
        
        if mol is None:
            tqdm.write(f"Warning: Could not read molecule {i+1}. Skipping.")
            continue

        drugbank_id = mol.GetProp("DRUGBANK_ID") if mol.HasProp("DRUGBANK_ID") else None

        if not drugbank_id:
            tqdm.write(f"Warning: Molecule {i+1} has no 'DRUGBANK_ID' property. Skipping.")
            continue

        # 1. Add Hydrogens
        mol_with_hs = Chem.AddHs(mol)

        # 2. Generate 3D Coordinates
        # Create an EmbedParameters object and set parameters on it
        params = AllChem.ETKDGv2()
        params.randomSeed = 42 # Adjust randomSeed as needed for reproducibility
        # You can add other parameters to 'params' as needed, e.g., params.maxAttempts = 1000

        embed_result = AllChem.EmbedMolecule(mol_with_hs, params) # Pass the parameters object

        if embed_result == -1: # -1 indicates failure to embed
            tqdm.write(f"Warning: Failed to generate 3D coordinates for '{drugbank_id}'. Skipping optimization and saving.")
            molecules_failed_3d += 1
            continue
        
        # 3. Optimize Geometry
        # mmffVariant='MMFF94' is default, can also be 'MMFF94s'
        # MMFFOptimizeMolecule returns 0 if converged, 1 if not converged
        opt_result = AllChem.MMFFOptimizeMolecule(mol_with_hs, maxIters=1000) # Increased iterations for better optimization

        if opt_result != 0:
            tqdm.write(f"Warning: Geometry optimization for '{drugbank_id}' did not converge (result: {opt_result}). Saving anyway.")
            molecules_failed_opt += 1

        # 4. (Optional) Remove Hydrogens before saving
        if remove_hydrogens_before_save:
            final_mol = Chem.RemoveHs(mol_with_hs)
        else:
            final_mol = mol_with_hs

        # 5. Save the molecule
        safe_drugbank_id = "".join(c for c in drugbank_id if c.isalnum() or c in ('-', '_', '.'))
        if not safe_drugbank_id:
             safe_drugbank_id = f"molecule_{i+1}_no_clean_id"

        output_filename = os.path.join(output_directory_path, f"{safe_drugbank_id}.sdf")
        writer = Chem.SDWriter(output_filename)
        writer.write(final_mol)
        writer.close()
        # tqdm.write(f"Saved molecule '{drugbank_id}' to '{output_filename}'") # Too verbose for large runs
        molecules_saved += 1

    print(f"\n--- Processing Summary ---")
    print(f"Total molecules read: {molecules_processed}")
    print(f"Molecules with DRUGBANK_ID saved: {molecules_saved}")
    print(f"Molecules that failed 3D embedding: {molecules_failed_3d}")
    print(f"Molecules where optimization did not converge: {molecules_failed_opt}")
    print("Script finished.")

if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage: python process_sdf.py <input_sdf_file_path> <output_directory_path> [remove_hydrogens_before_save]")
        print("  <input_sdf_file_path>: Full path to the input SDF file.")
        print("  <output_directory_path>: Full path to the directory where split SDF files will be saved.")
        print("  [remove_hydrogens_before_save]: Optional. Set to 'true' or 'yes' to remove hydrogens before saving. Default is to keep them.")
        print("Example: python process_sdf.py /home/user/data/drugbank.sdf /home/user/output/3d_drugs true")
        sys.exit(1)

    input_sdf = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Check for the optional argument to remove hydrogens
    remove_hs = False
    if len(sys.argv) == 4:
        if sys.argv[3].lower() in ['true', 'yes', 't', 'y']:
            remove_hs = True

    split_and_process_sdf(input_sdf, output_dir, remove_hs)