"""The two legacy identity stages must refuse their corrupted resolver.

``identify_required_structures.py`` and ``generate_negative_samples.py`` both
build ``dict(zip(uniprot_df['Mapped_Gene_Name'], uniprot_df['Entry']))`` --
last-duplicate-wins over a column in which 57.73% of rows name a different
protein. It collapses 4,585 accessions onto 1,795 symbols, discarding 2,790, and
the survivor of each collapse became the receptor that was docked: 633 of the
campaign's 979 proteins (64.66%) are not the protein their job name claims.

Both are superseded by stages that resolve on the Entrez key through
IdentityIndex. These tests assert the old path cannot run by accident, and that
the escape hatch still exists for anyone deliberately reproducing the old run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import generate_negative_samples as gns  # noqa: E402
import identify_required_structures as irs  # noqa: E402

STAGES = (
    (irs, "identify_required_structures.py",
     {"resolve_interactions.py", "plan_positives.py"}),
    (gns, "generate_negative_samples.py", {"sample_negatives.py"}),
)


class TestResolverRefusal(unittest.TestCase):
    def test_default_refuses(self):
        for module, name, _ in STAGES:
            with self.subTest(stage=name):
                with self.assertRaises(module.CorruptedResolverRefused):
                    module.assert_resolver_not_used()

    def test_refusal_names_its_replacement(self):
        """An error that does not say what to run instead just blocks the user."""
        for module, name, replacements in STAGES:
            with self.subTest(stage=name):
                try:
                    module.assert_resolver_not_used()
                except module.CorruptedResolverRefused as exc:
                    message = str(exc)
                for replacement in replacements:
                    self.assertIn(replacement, message)
                self.assertIn("57.73%", message)
                self.assertNotIn("57.73%%", message)  # no stray format escape

    def test_explicit_opt_in_is_allowed(self):
        for module, name, _ in STAGES:
            with self.subTest(stage=name):
                self.assertIsNone(module.assert_resolver_not_used(allow_legacy=True))

    def test_replacement_stages_are_declared_and_exist(self):
        for module, name, replacements in STAGES:
            with self.subTest(stage=name):
                self.assertEqual(set(module.REPLACEMENT_STAGES), replacements)
                for replacement in replacements:
                    self.assertTrue(
                        os.path.exists(os.path.join(REPO, replacement)),
                        "%s names %s as its replacement but that file does not exist"
                        % (name, replacement),
                    )


class TestExitCodes(unittest.TestCase):
    """Running either bare must FAIL, not quietly do the corrupted thing."""

    def _run(self, script, *args):
        return subprocess.run(
            [sys.executable, os.path.join(REPO, script), *args],
            cwd=REPO, capture_output=True, text=True, timeout=120,
        )

    def test_bare_invocation_exits_nonzero(self):
        for _, name, _ in STAGES:
            with self.subTest(stage=name):
                proc = self._run(name)
                self.assertEqual(
                    proc.returncode, 2,
                    "%s exited %d; a refusal that exits 0 reads as success in a "
                    "shell pipeline" % (name, proc.returncode),
                )

    def test_refusal_message_reaches_the_operator(self):
        for _, name, replacements in STAGES:
            with self.subTest(stage=name):
                proc = self._run(name)
                combined = proc.stdout + proc.stderr
                self.assertIn("superseded", combined)
                for replacement in replacements:
                    self.assertIn(replacement, combined)


if __name__ == "__main__":
    unittest.main()
