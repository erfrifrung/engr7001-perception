"""Regression checks for the V2 merged-mask suppression patch."""
import itertools

import numpy as np

from wp3_live.proposals import suppress_merged_masks


def test_union_is_suppressed_in_every_order():
    children = np.eye(3, dtype=bool).repeat(20, axis=1)
    merged = np.logical_or.reduce(children)
    for ordering in itertools.permutations([merged, *children]):
        kept = suppress_merged_masks(list(ordering))
        assert [mask.tobytes() for mask in kept] == [
            mask.tobytes() for mask in ordering if mask is not merged
        ]


def test_lone_or_insufficiently_explained_mask_is_retained():
    full = np.ones(100, dtype=bool)
    first = np.arange(100) < 30
    overlapping = (np.arange(100) >= 10) & (np.arange(100) < 50)
    other = np.arange(100) >= 70
    for masks in ([full], [full, first, overlapping], [full, first, other]):
        assert all(a is b for a, b in zip(suppress_merged_masks(masks), masks, strict=True))
