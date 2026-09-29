"""Real instrument background/noise sampled from recorded buffer movies."""

import numpy as np
from alex_area.movie_generator.buffer_movies import BufferMovie, load_buffer_movies

from simulator.constants import DEFAULT_MOV_THUMBNAIL_SIZE, VAL_MOV_THUMBNAIL_SIZE

B_MOV_X_MAX: int = 231
B_MOV_Y_MAX: int = 163
VAL_B_MOV_X_MAX: int = 145
VAL_B_MOV_Y_MAX: int = 56

_all_b_movs = load_buffer_movies()

# Buffer movies capture real instrument background/noise with no particle
# events. Indices 12+ are TwoMP large-FoV movies, used for training; indices
# 0-11 are a smaller-FoV instrument setup, held out entirely as an
# unseen-background validation set.
BUFFER_MOVIES = [
    mov[:, :B_MOV_Y_MAX, :B_MOV_X_MAX] for mov in _all_b_movs[12:]
]  # crop all buffer movies to a common (Y, X) footprint
VAL_BUFFER_MOVIES = [
    mov[:, :VAL_B_MOV_Y_MAX, :VAL_B_MOV_X_MAX] for mov in _all_b_movs[:12]
]  # crop all buffer movies to a common (Y, X) footprint


def gen_random_mov_stack(
    length: int = 500,
    mov_thumbnail_size: int | None = None,
    validation: bool = False,
    rng: np.random.Generator | None = None,
) -> BufferMovie:
    """Sample a random spatiotemporal crop from a randomly chosen buffer movie.

    Selects one of `BUFFER_MOVIES` (or `VAL_BUFFER_MOVIES` if `validation`)
    at random, then crops it to a `mov_thumbnail_size` x `mov_thumbnail_size`
    (y, x) window and a `length`-frame time window, each placed at a random
    offset. Used to vary the background content and temporal window seen
    during training/validation.

    Args:
        length: Number of frames to crop from the movie's time axis.
        mov_thumbnail_size: Width and height, in px, of the cropped spatial
            window. Defaults to `VAL_MOV_THUMBNAIL_SIZE` if `validation`,
            else `DEFAULT_MOV_THUMBNAIL_SIZE`.
        validation: If True, draws from `VAL_BUFFER_MOVIES` -- the smaller-
            FoV buffer movies held out entirely from training -- instead of
            `BUFFER_MOVIES`.
        rng: Random generator to sample from. Defaults to a fresh, unseeded
            `np.random.default_rng()` if not given.

    Returns:
        The cropped buffer movie, shape (length, mov_thumbnail_size,
        mov_thumbnail_size).
    """
    rng = np.random.default_rng(rng)
    if mov_thumbnail_size is None:
        mov_thumbnail_size = (
            VAL_MOV_THUMBNAIL_SIZE if validation else DEFAULT_MOV_THUMBNAIL_SIZE
        )

    movies = VAL_BUFFER_MOVIES if validation else BUFFER_MOVIES
    x_bound = VAL_B_MOV_X_MAX if validation else B_MOV_X_MAX
    y_bound = VAL_B_MOV_Y_MAX if validation else B_MOV_Y_MAX

    rand_index = rng.integers(0, len(movies))
    mov: BufferMovie = movies[rand_index]

    x_width = mov_thumbnail_size
    y_width = mov_thumbnail_size

    x_max = x_bound - x_width
    y_max = y_bound - y_width
    t_max = mov.shape[0] - length

    y_rand = rng.integers(0, y_max + 1)
    x_rand = rng.integers(0, x_max + 1)
    t_rand = rng.integers(0, t_max + 1)

    return mov[
        t_rand : t_rand + length, y_rand : y_rand + y_width, x_rand : x_rand + x_width
    ]
