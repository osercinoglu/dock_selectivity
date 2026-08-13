"""Every command-line argument must document itself.

You cannot run a pipeline whose flags you have to read the source to understand.
This walks the AST of every script and fails if any `add_argument` for a flag
lacks `help=`, so a new undocumented argument breaks the suite rather than
quietly landing.
"""

from __future__ import annotations

import ast
import os
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: gnina_remote.py's internal parser deliberately mimics the gnina binary's own
#: CLI (add_help=False, unknown flags passed through). Its "documentation" is
#: gnina's; adding argparse help there would imply it owns those flags.
EXEMPT = {"gnina_remote.py"}


def scripts():
    for name in sorted(os.listdir(REPO)):
        if name.endswith(".py") and name not in EXEMPT:
            yield name, os.path.join(REPO, name)
    package = os.path.join(REPO, "dock_selectivity")
    for name in sorted(os.listdir(package)):
        if name.endswith(".py"):
            yield "dock_selectivity/" + name, os.path.join(package, name)


class TestArgumentHelpCoverage(unittest.TestCase):
    def test_every_flag_has_help_text(self):
        undocumented = []
        for label, path in scripts():
            tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and getattr(node.func, "attr", None) == "add_argument"):
                    continue
                flag = next((a.value for a in node.args
                             if isinstance(a, ast.Constant)), None)
                if not (isinstance(flag, str) and flag.startswith("-")):
                    continue
                if not any(k.arg == "help" for k in node.keywords):
                    undocumented.append("%s:%d %s" % (label, node.lineno, flag))
        self.assertEqual(
            undocumented, [],
            "these arguments have no help text, so --help does not explain them: %s"
            % undocumented)

    def test_every_script_has_a_module_docstring(self):
        missing = []
        for label, path in scripts():
            tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
            doc = ast.get_docstring(tree)
            if not doc or len(doc.strip()) < 40:
                missing.append(label)
        self.assertEqual(missing, [],
                         "these have no usable module docstring: %s" % missing)
