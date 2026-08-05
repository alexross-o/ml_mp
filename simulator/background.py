"""Real instrument background/noise sampled from recorded buffer movies."""

import numpy as np
from alex_area.movie_generator.buffer_movies import BufferMovie, load_buffer_movies

B_MOV_X_MAX: int = 231
B_MOV_Y_MAX: int = 163

# Buffer movies capture real instrument background/noise with no particle events;
# indices 12+ correspond to TwoMP large-FoV buffer movies.
_b_movs = load_buffer_movies()[12:]
BUFFER_MOVIES = [
    mov[:, :B_MOV_Y_MAX, :B_MOV_X_MAX] for mov in _b_movs
]  # crop all buffer movies to a common (Y, X) footprint


def gen_random_mov_stack(length: int = 500, mov_thumbnail_size: int = 64) -> BufferMovie:
    """Sample a random spatiotemporal crop from a randomly chosen buffer movie.

    Selects one of `BUFFER_MOVIES` at random, then crops it to a
    `mov_thumbnail_size` x `mov_thumbnail_size` (y, x) window and a
    `length`-frame time window, each placed at a random offset. Used to vary
    the background content and temporal window seen during training.

    Args:
        length: Number of frames to crop from the movie's time axis.
        mov_thumbnail_size: Width and height, in px, of the cropped spatial
            window.

    Returns:
        The cropped buffer movie, shape (length, mov_thumbnail_size,
        mov_thumbnail_size).
    """
    rand_index = np.random.randint(0, len(BUFFER_MOVIES))
    mov: BufferMovie = BUFFER_MOVIES[rand_index]

    x_width = mov_thumbnail_size
    y_width = mov_thumbnail_size

    x_max = B_MOV_X_MAX - x_width
    y_max = B_MOV_Y_MAX - y_width
    t_max = mov.shape[0] - length

    y_rand = np.random.randint(0, y_max + 1)
    x_rand = np.random.randint(0, x_max + 1)
    t_rand = np.random.randint(0, t_max + 1)

    return mov[
        t_rand : t_rand + length, y_rand : y_rand + y_width, x_rand : x_rand + x_width
    ]
