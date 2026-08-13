"""G6 lint -- no free-text identifier resolution may reappear in the repo.

FIX_PLAN §1.4 guarantee G6. The original defect was::

    df = service.get_df(gene_name, columns="Entry,Entry Name,Gene Names,Organism")
    ...
    df_filtered['Mapped_Gene_Name'] = gene_name     # broadcast to EVERY row

A free-text *search* was used where an exact *lookup* was required, and the
queried name was then stamped onto every returned row. 57.73% of the resulting
mapping named a different protein, and because the accession it produced selected
the receptor, 633 of the campaign's 979 docked proteins were the wrong protein.

These tests fail if that pattern returns. They parse the AST rather than grepping
text, because the repo legitimately *describes* the old bug in docstrings and
comments -- a grep-based lint would either fire on the documentation or have to be
weakened until it caught nothing.
"""

from __future__ import annotations

import ast
import os
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Modules whose whole purpose is free-text identifier search.
BANNED_IMPORTS = {"bioservices"}

#: Attribute calls that perform a free-text identifier query.
BANNED_CALLS = {"get_df", "quick_search", "search_uniprot"}

#: The corrupted column. Nothing may read it as a resolution key ever again.
BANNED_NAMES = {"Mapped_Gene_Name"}


def production_files():
    """Every .py in the repo root, excluding tests."""
    out = []
    for name in sorted(os.listdir(REPO)):
        if name.endswith(".py") and not name.startswith("test_"):
            out.append(os.path.join(REPO, name))
    package = os.path.join(REPO, "dock_selectivity")
    if os.path.isdir(package):
        for name in sorted(os.listdir(package)):
            if name.endswith(".py"):
                out.append(os.path.join(package, name))
    return out


def parse(path):
    with open(path, encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=path)


class TestG6NoFreeTextResolution(unittest.TestCase):
    def test_the_corrupted_script_is_gone(self):
        self.assertFalse(
            os.path.exists(os.path.join(REPO, "prepare_uniprot_mapping.py")),
            "prepare_uniprot_mapping.py is retired: its free-text search, its "
            "hash-randomised list(set(...)) query order and its "
            "drop_duplicates(subset=['Entry']) together made the output both wrong "
            "and non-reproducible. build_identity_index.py replaces it.",
        )

    def test_no_module_imports_a_free_text_search_client(self):
        offenders = []
        for path in production_files():
            for node in ast.walk(parse(path)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in BANNED_IMPORTS:
                            offenders.append((path, node.lineno, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    if (node.module or "").split(".")[0] in BANNED_IMPORTS:
                        offenders.append((path, node.lineno, node.module))
        self.assertEqual(offenders, [], "free-text search client imported: %r" % offenders)

    def test_no_module_calls_a_free_text_query(self):
        offenders = []
        for path in production_files():
            for node in ast.walk(parse(path)):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = getattr(func, "attr", None) or getattr(func, "id", None)
                    if name in BANNED_CALLS:
                        offenders.append((os.path.basename(path), node.lineno, name))
        self.assertEqual(
            offenders, [],
            "free-text identifier query called (docstrings describing the old bug "
            "are fine; a real call is not): %r" % offenders,
        )

    def test_no_module_reads_the_corrupted_column_as_a_key(self):
        """A string constant in a docstring is prose; a subscript is a data path."""
        offenders = []
        for path in production_files():
            tree = parse(path)
            for node in ast.walk(tree):
                # df['Mapped_Gene_Name'] / df.get('Mapped_Gene_Name')
                if isinstance(node, ast.Subscript):
                    sub = node.slice
                    value = getattr(sub, "value", sub)
                    if isinstance(value, ast.Constant) and value.value in BANNED_NAMES:
                        offenders.append((os.path.basename(path), node.lineno))
        self.assertEqual(
            offenders, [],
            "Mapped_Gene_Name read as a data key -- that column is 57.73%% wrong "
            "and must never be a resolution input again: %r" % offenders,
        )

    def test_every_production_file_still_parses(self):
        """Cheap guard: pocketmatch_clustering.py once shipped a stray 'tmux' token."""
        for path in production_files():
            with self.subTest(file=os.path.basename(path)):
                parse(path)


if __name__ == "__main__":
    unittest.main()
