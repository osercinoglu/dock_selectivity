#!/usr/bin/env python3
"""Drop-in stand-in for the gnina binary that runs it on a remote GPU host.

Why a shim rather than a pipeline stage
---------------------------------------

``consensus_docker.py`` invokes gnina as one self-contained command::

    gnina -r <receptor.pdb> -l <ligand.sdf> \\
          --center_x/y/z --size_x/y/z -o <out.sdf> --log <out.log> --cpu N

and does everything else -- splitting ``out.sdf`` into per-pose files, building
``complex_*.pdb``, writing ``results.csv`` -- **locally, after gnina returns**. So
only that one call has to happen on the GPU box. This script presents the same
command line, ships the two inputs over SSH, runs gnina in a container on the
remote, and writes ``out.sdf`` and ``out.log`` back to the exact local paths it
was given.

Point ``GNINA_PATH`` at this file and nothing else changes::

    GNINA_PATH = "/home/onur/repos/dock_selectivity/gnina_remote.py"

``consensus_docker.py`` cannot tell the difference, and because each job pulls its
own results the moment gnina finishes, the pipeline's existing skip-if-complete
behaviour makes the remote stage resumable with no extra bookkeeping.

Transfer per job is ~300 KB up and ~1-2 MB down, against a gnina run of minutes.

Configuration (environment, so no secrets live in the repo)::

    GNINA_REMOTE_HOST   user@host          required
    GNINA_REMOTE_PORT   ssh port           default 22
    GNINA_REMOTE_IMAGE  container image    default gnina/gnina:v1.3.3
    GNINA_REMOTE_DIR    remote scratch     default ~/gnina_jobs
    GNINA_REMOTE_KEY    private key path   optional
    GNINA_REMOTE_NATIVE path to a native gnina binary on the remote; if set, the
                        container is not used

Authentication is SSH keys only. Passwords are deliberately unsupported: this runs
unattended once per job, and the alternatives (``sshpass``, a stored secret) put
the password in the process list or on disk.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
import uuid

DEFAULT_IMAGE = "gnina/gnina:v1.3.3"
DEFAULT_REMOTE_DIR = "gnina_jobs"

#: Flags taking a value that we must intercept because they name a local path.
_PATH_FLAGS = {"-r", "--receptor", "-l", "--ligand", "-o", "--out", "--log"}


class RemoteGninaError(RuntimeError):
    """Raised when the remote leg fails. Never swallowed into a fake success."""


def ssh_base(host, port, key):
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    if port:
        cmd += ["-p", str(port)]
    if key:
        cmd += ["-i", key]
    return cmd + [host]


def scp_base(port, key):
    cmd = ["scp", "-q", "-o", "BatchMode=yes"]
    if port:
        cmd += ["-P", str(port)]
    if key:
        cmd += ["-i", key]
    return cmd


def run(cmd, what, timeout=None):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RemoteGninaError(
            "%s failed (exit %d)\n  cmd: %s\n  stderr: %s"
            % (what, proc.returncode, " ".join(shlex.quote(c) for c in cmd),
               (proc.stderr or "").strip()[:2000])
        )
    return proc


def parse_args(argv):
    """Split gnina's command line into the paths we handle and the rest.

    Unknown flags pass through untouched, so a gnina option this shim has never
    heard of still reaches the binary.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-r", "--receptor")
    parser.add_argument("-l", "--ligand")
    parser.add_argument("-o", "--out")
    parser.add_argument("--log")
    known, rest = parser.parse_known_args(argv)
    return known, rest


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    known, passthrough = parse_args(argv)

    host = os.environ.get("GNINA_REMOTE_HOST")
    if not host:
        sys.stderr.write(
            "gnina_remote: GNINA_REMOTE_HOST is not set. This is a stand-in for the "
            "gnina binary that runs it on a remote GPU host; without a host there is "
            "nothing it can do, and silently falling back to a local gnina would hide "
            "that the CNN scores came from somewhere else.\n")
        return 2
    port = os.environ.get("GNINA_REMOTE_PORT")
    key = os.environ.get("GNINA_REMOTE_KEY")
    image = os.environ.get("GNINA_REMOTE_IMAGE", DEFAULT_IMAGE)
    native = os.environ.get("GNINA_REMOTE_NATIVE")
    remote_root = os.environ.get("GNINA_REMOTE_DIR", DEFAULT_REMOTE_DIR)
    # A quoted "~" is literal to the remote shell: shlex.quote("~/x") makes it
    # create a directory actually named "~". Keep the path relative (ssh resolves
    # it against $HOME) or absolute, never tilde-prefixed.
    if remote_root.startswith("~/"):
        remote_root = remote_root[2:]
    elif remote_root == "~":
        remote_root = ""

    for flag, path in (("-r/--receptor", known.receptor), ("-l/--ligand", known.ligand)):
        if not path:
            sys.stderr.write("gnina_remote: %s is required\n" % flag)
            return 2
        if not os.path.exists(path):
            sys.stderr.write("gnina_remote: %s not found: %s\n" % (flag, path))
            return 2
    if not known.out:
        sys.stderr.write("gnina_remote: -o/--out is required\n")
        return 2

    ssh = ssh_base(host, port, key)
    scp = scp_base(port, key)
    job = "job_%s_%d" % (uuid.uuid4().hex[:10], os.getpid())
    remote_dir = "%s/%s" % (remote_root.rstrip("/"), job)
    started = time.time()

    try:
        # Resolve to an ABSOLUTE remote path: `docker -v` rejects a relative one
        # (exit 125), and the tilde cannot be quoted through the remote shell.
        made = run(ssh + ["mkdir -p %s && cd %s && pwd" %
                          (shlex.quote(remote_dir), shlex.quote(remote_dir))],
                   "remote mkdir", timeout=120)
        remote_dir = made.stdout.strip().splitlines()[-1]

        target = "%s:%s/" % (host, remote_dir)
        run(scp + [known.receptor, known.ligand, target], "upload", timeout=900)

        r_name = os.path.basename(known.receptor)
        l_name = os.path.basename(known.ligand)

        if native:
            inner = [native, "-r", "%s/%s" % (remote_dir, r_name),
                     "-l", "%s/%s" % (remote_dir, l_name),
                     "-o", "%s/out.sdf" % remote_dir,
                     "--log", "%s/out.log" % remote_dir] + passthrough
            remote_cmd = " ".join(shlex.quote(c) for c in inner)
        else:
            inner = ["gnina", "-r", "/w/%s" % r_name, "-l", "/w/%s" % l_name,
                     "-o", "/w/out.sdf", "--log", "/w/out.log"] + passthrough
            remote_cmd = (
                "docker run --rm --gpus all -v %s:/w %s %s"
                % (shlex.quote(remote_dir), shlex.quote(image),
                   " ".join(shlex.quote(c) for c in inner))
            )

        proc = subprocess.run(ssh + [remote_cmd], capture_output=True, text=True)
        # gnina's own stdout/stderr belong to the caller verbatim: consensus_docker
        # parses and logs it, and hiding it would make a remote failure look local.
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            raise RemoteGninaError("remote gnina exited %d" % proc.returncode)

        os.makedirs(os.path.dirname(os.path.abspath(known.out)) or ".", exist_ok=True)
        run(scp + ["%s:%s/out.sdf" % (host, remote_dir), known.out],
            "download out.sdf", timeout=900)
        if known.log:
            os.makedirs(os.path.dirname(os.path.abspath(known.log)) or ".", exist_ok=True)
            try:
                run(scp + ["%s:%s/out.log" % (host, remote_dir), known.log],
                    "download out.log", timeout=300)
            except RemoteGninaError:
                pass  # the log is advisory; out.sdf is the result

        if not os.path.exists(known.out) or os.path.getsize(known.out) == 0:
            raise RemoteGninaError(
                "out.sdf is missing or empty after download: %s. Treating this as a "
                "failure rather than letting an empty pose file look like a result."
                % known.out)

        sys.stderr.write("gnina_remote: %s -> %d bytes in %.1fs\n"
                         % (host, os.path.getsize(known.out), time.time() - started))
        return 0

    except (RemoteGninaError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write("gnina_remote: %s\n" % exc)
        return 1
    finally:
        subprocess.run(ssh + ["rm -rf %s" % shlex.quote(remote_dir)],
                       capture_output=True, text=True)


if __name__ == "__main__":
    sys.exit(main())
