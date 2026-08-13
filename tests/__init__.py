"""Test suite for dock_selectivity.

Run with the stdlib runner (pytest is not installed in the ``teachopencadd``
environment)::

    python3 -m unittest discover -s tests -t .

from the repository root, or with absolute paths::

    python3 -m unittest discover \\
        -s /home/onur/repos/dock_selectivity/tests \\
        -t /home/onur/repos/dock_selectivity

Every test here is **hermetic**: no network, no external drive, no ``/opt/data``.
Fixtures are built in ``tempfile.TemporaryDirectory()``.  Real values copied out
of the campaign (accessions, gene symbols, directory names) appear as literals,
never as reads.
"""
