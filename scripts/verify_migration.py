#!/usr/bin/env python3
"""Run the four migration gates from the project root using the launcher."""

import sys
import unittest

from yam_bag_rl.paths import PROJECT_ROOT


GATES = (
    ("Zero legacy dependency", "test_zero_old_repo_dep.py", "tests", 2),
    ("Yam_B assets, FK, controller and static gravity", "test_robot_assets.py", "tests", 2),
    ("Monday dual-tether trajectory equivalence", "test_monday_rope_equiv.py", "tests", 2),
    ("Policy V3 baseline", "test_*.py", "tests/policy_v3", 16),
)


def main():
    passed = 0
    for name, pattern, directory, expected in GATES:
        print(f"\n{name}", flush=True)
        tests = unittest.TestLoader().discover(str(PROJECT_ROOT / directory), pattern, top_level_dir=str(PROJECT_ROOT))
        result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(tests)
        ok = result.testsRun == expected and result.wasSuccessful() and not result.skipped
        print(f"{'PASS' if ok else 'FAIL'} {name} ({result.testsRun} tests)", flush=True)
        passed += ok
    print(f"\n{passed}/{len(GATES)} migration gates passed", flush=True)
    return 0 if passed == len(GATES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
