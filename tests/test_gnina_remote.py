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


class TestTransientClassification(unittest.TestCase):
    """Retry the network; never retry a real error.

    Retrying a permanent failure wastes minutes per job across thousands of jobs
    and buries the message that would have explained it.
    """

    def test_network_failures_are_transient(self):
        for text in ("ssh: connect to host x port 50000: Connection refused",
                     "Connection timed out",
                     "client_loop: send disconnect: Broken pipe",
                     "ssh: Could not resolve hostname x: Temporary failure in name resolution",
                     "kex_exchange_identification: read: Connection reset by peer",
                     "Network is unreachable"):
            with self.subTest(text=text):
                self.assertTrue(gr.is_transient(255, text), text)

    def test_ssh_exit_255_alone_is_transient(self):
        self.assertTrue(gr.is_transient(255, ""))

    def test_real_errors_are_not_transient(self):
        for code, text in ((1, "gnina: unrecognised option '--nope'"),
                           (125, "docker: invalid reference format"),
                           (1, "scp: /x/y: No such file or directory"),
                           (2, "Permission denied (publickey)")):
            with self.subTest(text=text):
                self.assertFalse(gr.is_transient(code, text), text)


class TestRetryBehaviour(unittest.TestCase):
    def test_a_permanent_failure_is_raised_immediately(self):
        """One attempt, not five, and the original stderr survives."""
        with self.assertRaises(gr.RemoteGninaError) as ctx:
            gr.run_resilient(["sh", "-c", "echo 'No such file or directory' >&2; exit 1"],
                             "probe", retries=5)
        self.assertNotIsInstance(ctx.exception, gr.TransientError)
        self.assertIn("No such file", str(ctx.exception))

    def test_a_transient_failure_is_retried_then_gives_up(self):
        seen = []
        original = gr.time.sleep
        gr.time.sleep = lambda s: seen.append(s)        # no real waiting
        try:
            with self.assertRaises(gr.TransientError):
                gr.run_resilient(
                    ["sh", "-c", "echo 'Connection refused' >&2; exit 255"],
                    "probe", retries=3)
        finally:
            gr.time.sleep = original
        self.assertEqual(len(seen), 2, "3 attempts means 2 waits")
        self.assertGreater(seen[1], seen[0], "backoff must grow")

    def test_success_returns_without_sleeping(self):
        original = gr.time.sleep
        gr.time.sleep = lambda s: self.fail("must not sleep on success")
        try:
            proc = gr.run_resilient(["sh", "-c", "echo ok"], "probe", retries=5)
        finally:
            gr.time.sleep = original
        self.assertEqual(proc.stdout.strip(), "ok")


class TestContentAddressedResume(unittest.TestCase):
    """Same inputs -> same remote directory, so a re-run collects instead of re-docking."""

    def _pair(self, tmp, receptor=b"REC", ligand=b"LIG"):
        r = os.path.join(tmp, "r.pdb"); open(r, "wb").write(receptor)
        l = os.path.join(tmp, "l.sdf"); open(l, "wb").write(ligand)
        return r, l

    def test_identical_inputs_give_the_same_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, l = self._pair(tmp)
            self.assertEqual(gr.job_key(r, l, ["--seed", "42"]),
                             gr.job_key(r, l, ["--seed", "42"]))

    def test_different_arguments_give_a_different_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, l = self._pair(tmp)
            self.assertNotEqual(gr.job_key(r, l, ["--seed", "42"]),
                                gr.job_key(r, l, ["--seed", "99"]))

    def test_different_content_gives_a_different_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            r1, l1 = self._pair(tmp)
            r2 = os.path.join(tmp, "r2.pdb"); open(r2, "wb").write(b"OTHER")
            self.assertNotEqual(gr.job_key(r1, l1, []), gr.job_key(r2, l1, []))


class TestDetachedLaunch(unittest.TestCase):
    """The job must outlive its SSH session, or an outage destroys the run."""

    def test_launch_command_detaches_and_records_its_exit_code(self):
        cmd = gr.build_remote_command("/abs/dir", "r.pdb", "l.sdf",
                                      ["--seed", "42"], "img:tag", None)
        self.assertIn("setsid", cmd)
        self.assertIn("nohup", cmd)
        self.assertIn("exit_code", cmd)
        self.assertTrue(cmd.rstrip().endswith("echo launched"))
        self.assertIn("&", cmd)

    def test_container_mount_uses_an_absolute_path(self):
        """docker -v rejects a relative path with exit 125."""
        cmd = gr.build_remote_command("/abs/dir", "r.pdb", "l.sdf", [], "img:tag", None)
        self.assertIn("-v /abs/dir:/w", cmd.replace("'", ""))

    def test_native_mode_skips_the_container(self):
        cmd = gr.build_remote_command("/abs/dir", "r.pdb", "l.sdf", [],
                                      "img:tag", "/opt/gnina")
        self.assertNotIn("docker", cmd)
        self.assertIn("/opt/gnina", cmd)

    def test_passthrough_flags_reach_the_binary(self):
        cmd = gr.build_remote_command("/d", "r.pdb", "l.sdf",
                                      ["--exhaustiveness", "32"], "img", None)
        self.assertIn("--exhaustiveness", cmd)


class TestKeepAlives(unittest.TestCase):
    def test_ssh_sets_keepalives(self):
        """A quiet multi-minute run must not be culled by a NAT idle timeout."""
        joined = " ".join(gr.ssh_base("u@h", None, None))
        self.assertIn("ServerAliveInterval", joined)
        self.assertIn("ServerAliveCountMax", joined)


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
