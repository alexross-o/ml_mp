"""Ratiometric (contrast) conversion of a raw composited movie."""

import numpy as np


def gen_ratiometric_movie(movie: np.ndarray, navg: int) -> np.ndarray:
    """Convert a raw movie to ratiometric contrast frames.

    At each valid center frame, sums the `navg` frames immediately before it
    (trailing) and the `navg` frames immediately after it (running) --
    excluding the center frame itself from both -- then computes
    `1 - running / trailing`. Frames too close to either end of the movie to
    have a full `navg`-frame window on both sides are set to NaN, exactly
    `navg` frames at each end -- matching the exclusion zone `gen_events`'
    `navg` argument keeps events out of.

    Args:
        movie: Raw movie, shape (T, H, W).
        navg: Number of frames summed into each trailing/running window.

    Returns:
        Ratiometric movie, same shape as `movie`, dtype float32. The first
        and last `navg` frames are NaN.

    Raises:
        ValueError: If `movie` has fewer than `2 * navg + 1` frames.
    """
    t_size = movie.shape[0]
    if t_size < 2 * navg + 1:
        raise ValueError("movie is too short for the given navg")

    centers = np.arange(navg, t_size - navg)

    # padded_cumsum[k] == movie[:k].sum(axis=0), so any window's sum is a
    # single subtraction instead of re-summing overlapping frames per center
    padded_cumsum = np.concatenate(
        [
            np.zeros((1, *movie.shape[1:]), dtype=np.int32),
            np.cumsum(movie, axis=0, dtype=np.int32),
        ]
    )

    trailing = padded_cumsum[centers] - padded_cumsum[centers - navg]
    running = padded_cumsum[centers + navg + 1] - padded_cumsum[centers + 1]

    eps = np.finfo(np.float32).eps
    ratio = 1.0 - running / (trailing + eps)

    ratiometric = np.full(movie.shape, np.nan, dtype=np.float32)
    ratiometric[centers] = ratio.astype(np.float32)

    return ratiometric
