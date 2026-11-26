"""Check both bundled Monday recordings and their vertical tether loads."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from yam_bag_rl.bag import proxy_contact, rope_replay
from yam_bag_rl.bag.lumped_contact_proxy import LumpedContactProxy

FIXTURES = Path(__file__).resolve().parent / "fixtures"
METRICS = json.loads((FIXTURES / "monday_metrics.json").read_text())
ATOL = 1e-6


class MondayReplayTest(unittest.TestCase):
    def check_case(self, case, replay):
        forces = []

        class MeasuredProxy(LumpedContactProxy):
            def step_with_reference(self, state, inputs, reference, inverse_mass=None):
                contact, length = super().step_with_reference(state, inputs, reference, inverse_mass)
                forces.append(contact.force[0].detach().numpy().copy())
                return contact, length

        with TemporaryDirectory() as temporary:
            result = Path(temporary) / "replayed.npz"
            with redirect_stdout(StringIO()):
                replay(FIXTURES / f"{case}_input.npz", result, proxy_type=MeasuredProxy)
            with (np.load(result) as actual, np.load(FIXTURES / f"{case}_expected.npz") as expected):
                self.assertEqual(set(actual.files) - {"steps", "wall_s"}, set(expected.files))
                max_abs = 0.
                for key in expected.files:
                    a, b = actual[key], expected[key]
                    with self.subTest(channel=key):
                        self.assertEqual(a.shape, b.shape)
                        self.assertEqual(a.dtype, b.dtype)
                        if b.dtype.kind in "biu":
                            np.testing.assert_array_equal(a, b)
                        else:
                            np.testing.assert_allclose(a, b, rtol=0., atol=ATOL, equal_nan=False)
                        max_abs = max(max_abs, float(np.max(np.abs(a.astype(float) - b.astype(float)))))
                n_arrays = len(expected.files)
                history = actual[METRICS[case]["history_key"]]

        self.assertEqual(len(history), 4800)
        self.assertEqual(len(forces), len(history))
        hover = (history[:, 0] >= 7.5) & (history[:, 0] < 8.5)
        self.assertEqual(int(hover.sum()), 500)
        self.assertTrue((history[hover, 1] > .001).all())
        self.assertTrue((history[hover, 2] <= ATOL).all())
        force = np.stack(forces)[hover, :, 2].sum(axis=1).mean().item()
        clearance = history[:, 1].max().item()
        np.testing.assert_allclose([force, clearance], [METRICS[case]["hover_vertical_force_N"],
                                   METRICS[case]["peak_clearance_m"]], rtol=0., atol=ATOL)
        print(f"{case}: {n_arrays} arrays, max |Δ|={max_abs:.3g}, "
              f"hover={force:.9f} N, peak={clearance * 1000:.6f} mm", flush=True)

    def test_neck_98mm(self):
        self.check_case("neck_98mm", rope_replay.run)

    def test_center_250g(self):
        self.check_case("center_250g", proxy_contact.run)
