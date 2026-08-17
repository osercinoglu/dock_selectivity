#!/usr/bin/env python3
"""Split a multi-molecule SDF into one 3D-prepared file per DrugBank ID.

Reads a combined SDF (e.g. DrugBank's approved-structures export), and for each
molecule adds hydrogens, embeds 3D coordinates, optimises the geometry and writes
``<DRUGBANK_ID>.sdf`` into the output folder. That folder is what the pipeline
passes as ``--ligands``: every docking job looks up its ligand by DrugBank ID, so
a molecule with no ``DRUGBANK_ID`` property is skipped and reported rather than
written under a guessed name.

Run once per ligand set, ahead of the pipeline.

The embedding seed and the optimisation iteration cap are flags, not literals:
they determine the geometry of every ligand that every downstream docking pose is
built from, so a run must be able to state (and reproduce) the values it used.

Usage::

    python3 split_sdf.py --in-sdf /opt/data/drugbank/approved_structures.sdf \
        --out /opt/data/drugbank/drugbank_approved_split

    # reproducibility knobs and a notification when it finishes
    python3 split_sdf.py --in-sdf drugbank.sdf --out ligands \
        --seed 42 --max-iters 1000 --remove-hydrogens --notify --notify-on all

The ``--out`` folder is what you then hand to the pipeline as ``--ligands``.
"""

import argparse
import logging
import os

from rdkit import Chem
from rdkit.Chem import AllChem  # Contains functions for 3D operations
from tqdm import tqdm

from dock_selectivity.runtime import add_runtime_arguments, resolve_out_dir, step

STEP_NAME = "split_sdf"

# --- Configuration ------------------------------------------------------------
# Defaults only: both are flags. They are the inputs to every ligand geometry the
# campaign docks, so they belong in the run's logged configuration.
DEFAULT_SEED = 42
DEFAULT_MAX_ITERS = 1000

# --- Implementation detail (the pipeline's naming contract, not configuration) -
# Docking jobs look their ligand up as <DRUGBANK_ID>.sdf, so this property name
# is fixed by the job-folder contract rather than chosen per run.
ID_PROPERTY = "DRUGBANK_ID"

LOG = logging.getLogger(STEP_NAME)


def split_and_process_sdf(input_sdf_file_path, output_directory_path,
                          remove_hydrogens_before_save=False,
                          seed=DEFAULT_SEED, max_iters=DEFAULT_MAX_ITERS):
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
        seed (int): ETKDG random seed; identical input plus identical seed gives
                    identical coordinates.
        max_iters (int): MMFF optimisation iteration cap.

    Returns:
        dict: counts of molecules read, saved, failed to embed and unconverged.
    """
    if not os.path.exists(input_sdf_file_path):
        raise FileNotFoundError(
            "input SDF file not found: {0}".format(input_sdf_file_path))

    os.makedirs(output_directory_path, exist_ok=True)

    LOG.info("Processing input file: %s", input_sdf_file_path)
    LOG.info("Saving processed molecules to: %s", output_directory_path)
    LOG.info("Remove hydrogens before saving: %s", remove_hydrogens_before_save)
    LOG.info("ETKDG seed: %s, MMFF max iterations: %s", seed, max_iters)

    supplier = Chem.SDMolSupplier(input_sdf_file_path)

    molecules_processed = 0
    molecules_saved = 0
    molecules_failed_3d = 0
    molecules_failed_opt = 0

    for i, mol in enumerate(tqdm(supplier, desc="Processing Molecules", unit="mol")):
        molecules_processed += 1

        if mol is None:
            LOG.warning("Could not read molecule %d. Skipping.", i + 1)
            continue

        drugbank_id = mol.GetProp(ID_PROPERTY) if mol.HasProp(ID_PROPERTY) else None

        if not drugbank_id:
            LOG.warning("Molecule %d has no '%s' property. Skipping.", i + 1, ID_PROPERTY)
            continue

        # 1. Add Hydrogens
        mol_with_hs = Chem.AddHs(mol)

        # 2. Generate 3D Coordinates
        # Create an EmbedParameters object and set parameters on it
        params = AllChem.ETKDGv2()
        params.randomSeed = seed

        embed_result = AllChem.EmbedMolecule(mol_with_hs, params)  # Pass the parameters object

        if embed_result == -1:  # -1 indicates failure to embed
            LOG.warning("Failed to generate 3D coordinates for '%s'. "
                        "Skipping optimization and saving.", drugbank_id)
            molecules_failed_3d += 1
            continue

        # 3. Optimize Geometry
        # mmffVariant='MMFF94' is default, can also be 'MMFF94s'
        # MMFFOptimizeMolecule returns 0 if converged, 1 if not converged
        opt_result = AllChem.MMFFOptimizeMolecule(mol_with_hs, maxIters=max_iters)

        if opt_result != 0:
            LOG.warning("Geometry optimization for '%s' did not converge "
                        "(result: %s). Saving anyway.", drugbank_id, opt_result)
            molecules_failed_opt += 1

        # 4. (Optional) Remove Hydrogens before saving
        if remove_hydrogens_before_save:
            final_mol = Chem.RemoveHs(mol_with_hs)
        else:
            final_mol = mol_with_hs

        # 5. Save the molecule
        safe_drugbank_id = "".join(c for c in drugbank_id if c.isalnum() or c in ('-', '_', '.'))
        if not safe_drugbank_id:
            safe_drugbank_id = "molecule_{0}_no_clean_id".format(i + 1)

        output_filename = os.path.join(output_directory_path, "{0}.sdf".format(safe_drugbank_id))
        writer = Chem.SDWriter(output_filename)
        writer.write(final_mol)
        writer.close()
        molecules_saved += 1

    LOG.info("--- Processing Summary ---")
    LOG.info("Total molecules read: %d", molecules_processed)
    LOG.info("Molecules with %s saved: %d", ID_PROPERTY, molecules_saved)
    LOG.info("Molecules that failed 3D embedding: %d", molecules_failed_3d)
    LOG.info("Molecules where optimization did not converge: %d", molecules_failed_opt)

    return {
        "molecules_read": molecules_processed,
        "molecules_saved": molecules_saved,
        "failed_3d": molecules_failed_3d,
        "unconverged": molecules_failed_opt,
    }


def build_parser():
    """Build the command-line parser for the ligand-splitting stage."""
    parser = argparse.ArgumentParser(
        prog="split_sdf.py",
        description=("Split one multi-molecule SDF into a folder of 3D-prepared "
                     "<DRUGBANK_ID>.sdf files -- the folder the pipeline takes as "
                     "--ligands."))
    parser.add_argument('--in-sdf', required=True,
                        help='combined input SDF (e.g. DrugBank approved structures). MANDATORY')
    parser.add_argument('--out', required=True,
                        help='directory to write one <DRUGBANK_ID>.sdf per molecule; '
                             'this is the folder the pipeline passes as --ligands. MANDATORY')
    parser.add_argument('--remove-hydrogens', action='store_true',
                        help='strip the hydrogens added for 3D embedding before saving '
                             '(default: keep them, which is what docking wants)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='ETKDG embedding random seed; the same input and seed give the '
                             'same coordinates (default: %(default)s)')
    parser.add_argument('--max-iters', type=int, default=DEFAULT_MAX_ITERS,
                        help='MMFF geometry-optimisation iteration cap (default: %(default)s)')
    parser.add_argument('--run-dir', default=None,
                        help='campaign run directory; when given, this step logs to '
                             '<run-dir>/steps/' + STEP_NAME + '/ (default: ./' + STEP_NAME + ')')
    return add_runtime_arguments(parser, STEP_NAME)


def main(argv=None):
    """Split the input SDF into per-ligand files; returns a process exit code."""
    args = build_parser().parse_args(argv)
    resolve_out_dir(args, STEP_NAME, args.run_dir)

    try:
        with step(STEP_NAME, args) as ctx:
            if not os.path.exists(args.in_sdf):
                ctx.logger.error("input SDF file not found: %s", args.in_sdf)
                raise SystemExit(2)
            summary = split_and_process_sdf(args.in_sdf, args.out,
                                            remove_hydrogens_before_save=args.remove_hydrogens,
                                            seed=args.seed, max_iters=args.max_iters)
            if summary["molecules_saved"] == 0:
                # An empty ligand folder is accepted silently by every downstream
                # stage and shows up only as "no jobs planned" much later.
                ctx.logger.error("no molecules were written to %s; the ligand folder "
                                 "would be empty", args.out)
                raise SystemExit(1)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
