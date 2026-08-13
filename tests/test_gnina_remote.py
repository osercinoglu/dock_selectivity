"""Tests for gnina_remote.py, the drop-in remote-GPU stand-in for gnina.

Hermetic: no SSH, no remote host. What is tested is the contract that makes the
shim a drop-in -- it must accept gnina's command line, refuse rather than
improvise when it cannot do its job, and never let a failed remote leg look like
a successful local one.

The live path is exercised separately by the pilot in SETUP.md.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import gnina_remote as gr  # noqa: E402


class TestCliCompatibility(unittest.TestCase):
    """consensus_docker.py builds exactly one command; the shim must accept it."""

    def test_the_real_consensus_docker_command_parses(self):
        argv = [
            "-r", "/x/receptor.pdb", "-l", "/x/ligand.sdf",
            "--center_x", "1.4", "--center_y", "1.09", "--center_z", "-11.53",
            "--size_x", "33.0", "--size_y", "32.0", "--size_z", "33.5",
            "-o", "/x/out.sdf", "--log", "/x/out.log",
            "--cpu", "60", "--num_modes", "20", "--exhaustiveness", "32",
        ]
        known, rest = gr.parse_args(argv)
        self.assertEqual(known.receptor, "/x/receptor.pdb")
        self.assertEqual(known.ligand, "/x/ligand.sdf")
        self.assertEqual(known.out, "/x/out.sdf")
        self.assertEqual(known.log, "/x/out.log")
        for flag in ("--center_x", "--size_z", "--cpu", "--num_modes", "--exhaustiveness"):
            self.assertIn(flag, rest, "%s must pass through to gnina untouched" % flag)

    def test_unknown_flags_pass_through(self):
        """A gnina option this shim never heard of must still reach the binary."""
        _, rest = gr.parse_args(["-r", "a", "-l", "b", "-o", "c", "--seed", "42",
                                 "--cnn_scoring", "rescore"])
        self.assertIn("--seed", rest)
        self.assertIn("--cnn_scoring", rest)

    def test_long_form_flags_are_equivalent(self):
        known, _ = gr.parse_args(["--receptor", "r.pdb", "--ligand", "l.sdf",
                                  "--out", "o.sdf"])
        self.assertEqual((known.receptor, known.ligand, known.out),
                         ("r.pdb", "l.sdf", "o.sdf"))


class TestRefusals(unittest.TestCase):
    """It must fail loudly, never improvise."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("GNINA_REMOTE_HOST", "GNINA_REMOTE_PORT", "GNINA_REMOTE_KEY")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_no_host_configured_exits_two(self):
        """Silently running a local gnina would hide where the CNN scores came from."""
        rc = gr.main(["-r", "r.pdb", "-l", "l.sdf", "-o", "o.sdf"])
        self.assertEqual(rc, 2)

    def test_missing_input_file_exits_two(self):
        os.environ["GNINA_REMOTE_HOST"] = "user@nowhere.invalid"
        with tempfile.TemporaryDirectory() as tmp:
            rc = gr.main(["-r", os.path.join(tmp, "absent.pdb"),
                          "-l", os.path.join(tmp, "absent.sdf"),
                          "-o", os.path.join(tmp, "o.sdf")])
            self.assertEqual(rc, 2)

    def test_missing_out_flag_exits_two(self):
        os.environ["GNINA_REMOTE_HOST"] = "user@nowhere.invalid"
        with tempfile.TemporaryDirectory() as tmp:
            r = os.path.join(tmp, "r.pdb"); open(r, "w").close()
            l = os.path.join(tmp, "l.sdf"); open(l, "w").close()
            self.assertEqual(gr.main(["-r", r, "-l", l]), 2)


class TestRemotePathHandling(unittest.TestCase):
    def test_tilde_is_stripped_not_quoted(self):
        """shlex.quote('~/x') is literal to the remote shell: it made a dir named '~'."""
        self.assertTrue(gr.DEFAULT_REMOTE_DIR and not gr.DEFAULT_REMOTE_DIR.startswith("~"))

    def test_ssh_and_scp_use_the_right_port_flag(self):
        """ssh takes -p, scp takes -P. Mixing them is a silent connection refusal."""
        self.assertIn("-p", gr.ssh_base("u@h", 50000, None))
        self.assertIn("-P", gr.scp_base(50000, None))
        self.assertIn("50000", gr.ssh_base("u@h", 50000, None))

    def test_batch_mode_is_forced(self):
        """No interactive password prompts in an unattended per-job loop."""
        self.assertIn("BatchMode=yes", " ".join(gr.ssh_base("u@h", None, None)))
        self.assertIn("BatchMode=yes", " ".join(gr.scp_base(None, None)))


if __name__ == "__main__":
    unittest.main()
