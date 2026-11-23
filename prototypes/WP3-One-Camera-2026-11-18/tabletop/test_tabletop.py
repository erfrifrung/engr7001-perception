"""Regression checks for flush joins, yaw conventions and highest-surface occlusion."""
import unittest

import numpy as np

from tabletop import CELL, cell_xy, footprint, grid_regions, split_region


class TabletopChecks(unittest.TestCase):
    def test_flush_prints_require_visible_seam(self):
        rows, cols = np.mgrid[10:48, 10:36]
        cells = np.column_stack((cols.ravel(), rows.ravel()))
        dims = np.array([0.075, 0.025, 0.015])
        gray = np.full((60, 60), 160, np.float32)
        lower = np.zeros(2)
        self.assertTrue(split_region(cells, gray, lower, dims)[0]["ambiguous"])
        self.assertEqual(len(split_region(cells, gray, lower, dims)), 1)
        gray[10:48, 22:24] = 80
        for points, surface, yaw in ((cells, gray, 90), (cells[:, ::-1], gray.T.copy(), 0)):
            children = split_region(points, surface, lower, dims)
            self.assertEqual(len(children), 2)
            for child in children:
                self.assertFalse(child["ambiguous"])
                self.assertAlmostEqual(child["yaw_deg_mod180"], yaw)
        # The same texture on a single footprint cannot create an extra instance.
        self.assertEqual(len(split_region(cells[cells[:, 0] < 23], gray, lower, dims)), 1)

    def test_near_square_orientation_is_unobservable(self):
        rows, cols = np.mgrid[10:34, 10:34]
        xy = cell_xy(np.column_stack((cols.ravel(), rows.ravel())), np.zeros(2))
        self.assertIsNone(footprint(xy, [0.047, 0.047, 0.038])["yaw_deg_mod180"])

    def test_taller_samples_occlude_lower_returns(self):
        xy = np.array([[[0.001, 0.001], [0.001, 0.001], [0.003, 0.001]]])
        height = np.array([[0.015, 0.070, 0.015]])
        bounds = np.array([[0, 0], [0.008, 0.008]])
        grid, _ = grid_regions(height, np.ones_like(height, bool), xy, bounds, [0.075, 0.025, 0.015], CELL**2)
        self.assertAlmostEqual(grid[0, 0], 0.070)
        self.assertAlmostEqual(grid[0, 1], 0.015)
        self.assertTrue(np.isneginf(grid[1, 1]))


if __name__ == "__main__":
    unittest.main()
