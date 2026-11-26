"""Check the standalone launcher and import origins without opening sibling projects."""

import ast
import importlib
import os
from pathlib import Path
import pkgutil
import subprocess
import sys
import unittest

import yam_bag_rl

ROOT = Path(__file__).resolve().parents[1]
FORMER_ROOT = Path("/home/ubuntu/workspace/gzl") / ("rando" + "mization")
LEGACY_IMPORTS = {"src", "experience", "yam_platform", "bag_sim", "policy_v3", "low_order_parcel"}


@unittest.skipUnless((ROOT / ".venv/bin/python").exists(), "Optional local Python environment")
def test_launcher_and_imports_stay_in_checkout():
    probe = """
import importlib
import os
from pathlib import Path
import sys

root = Path(os.environ["YAM_TEST_ROOT"])
assert Path(sys.executable).parent == root / ".venv/bin"
assert os.environ["PYTHONPATH"] == str(root / "src")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    assert os.environ[key] == "1"
for name in ("yam_bag_rl", "yam_bag_rl.bag.proxy_contact", "yam_bag_rl.robot.yam_model",
             "yam_bag_rl.policy_v3.training", "mujoco_warp", "torch", "warp"):
    module_file = Path(importlib.import_module(name).__file__).resolve()
    assert module_file.is_relative_to(root), (name, module_file)
for path in sys.path:
    assert not Path(path).is_relative_to(root.parent / ("rando" + "mization"))
"""
    subprocess.run([str(ROOT / "scripts/run_py.sh"), "--local", "-c", probe], cwd=ROOT,
                   env=os.environ | {"YAM_TEST_ROOT": str(ROOT)}, check=True)


def test_runtime_sources_have_no_sibling_checkout_path():
    forbidden = b"/" + b"rando" + b"mization/"
    runtime = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.sh"))
    for path in runtime:
        assert forbidden not in path.read_bytes(), path


class ZeroLegacyDependencyTest(unittest.TestCase):
    def test_all_modules_import_with_local_pythonpath(self):
        self.assertEqual(Path.cwd(), ROOT)
        self.assertEqual(Path(sys.executable).parent.parent, ROOT / ".venv")
        self.assertEqual(os.environ.get("PYTHONPATH"), str(ROOT / "src"))
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            self.assertEqual(os.environ.get(name), "1")

        imported = []
        for module in pkgutil.walk_packages(yam_bag_rl.__path__, "yam_bag_rl."):
            imported.append(importlib.import_module(module.name))
        self.assertTrue(all(any(m.__name__.startswith(f"yam_bag_rl.{name}.") for m in imported)
                            for name in ("robot", "bag", "policy_v3")))
        self.assertTrue(all(Path(m.__file__).resolve().is_relative_to(ROOT / "src") for m in imported))
        for entry in sys.path:
            path = Path(entry).resolve()
            self.assertFalse(path.is_relative_to(FORMER_ROOT), path)
        unexpected = []
        for name, module in sys.modules.copy().items():
            origin = getattr(module, "__file__", None)
            if origin:
                path = Path(origin).resolve()
                if path.is_relative_to(FORMER_ROOT):
                    unexpected.append((name, str(path)))
        self.assertEqual(unexpected, [])
        print(f"Imported {len(imported)} local modules with no former-source imports", flush=True)

    def test_local_source_and_resource_paths(self):
        bad = []
        scanned = 0
        old_segment = "/" + FORMER_ROOT.name + "/"
        for directory, children, files in os.walk(ROOT):
            # The Git internals and an optional external Python environment are
            # not part of the checked-out source or the local asset packages.
            children[:] = sorted(set(children) - {".git", ".venv"})
            for entry in files + children:
                path = Path(directory) / entry
                relative = path.relative_to(ROOT)
                if path.is_symlink() and path.resolve().is_relative_to(FORMER_ROOT):
                    bad.append(str(relative))
                if path.suffix not in {".py", ".xml", ".yaml", ".json"} or not path.is_file():
                    continue
                scanned += 1
                source = path.read_text()
                if str(FORMER_ROOT) in source or old_segment in source.replace("\\", "/"):
                    bad.append(str(relative))
                if path.suffix != ".py":
                    continue
                for node in ast.walk(ast.parse(source, filename=str(path))):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0:
                        names = [node.module]
                    else:
                        continue
                    if any(name.split(".")[0] in LEGACY_IMPORTS for name in names):
                        bad.append(f"{relative}:{node.lineno}")
        self.assertEqual(bad, [])
        print(f"Scanned {scanned} Python, XML, YAML and JSON files", flush=True)
