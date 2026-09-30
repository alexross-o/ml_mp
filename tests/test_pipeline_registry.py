"""Regression test for the Step 5 registry refactor (simulator/pipeline.py).

Checks that gen_data's returned ground truth heatmap has
len(EVENT_TYPES) channels rather than a hardcoded 3. One-time migration
safety net -- safe to delete once the refactor is trusted -- not a test
of ongoing behavior (a future 4th event type is expected to change this
channel count).
"""

from simulator.events import EVENT_TYPES
from simulator.pipeline import gen_data


def test_gen_data_heatmap_channel_count_matches_registry():
    movies, ground_truth = gen_data(batch_size=1, length=60, seed=0)

    assert movies.shape[0] == 1
    assert ground_truth["heatmap"].shape[0] == 1
    assert ground_truth["heatmap"].shape[1] == len(EVENT_TYPES)
    assert ground_truth["offset"].shape[1] == 2
    assert ground_truth["orientation"].shape[1] == 2
