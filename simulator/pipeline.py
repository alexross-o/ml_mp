"""PSF/instrument setup, mass calibration, and the end-to-end gen_data() pipeline."""

from functools import partial
from pathlib import Path

import numpy as np
import torch
from alex_area import utils
from alex_area.movie_generator.sim_movie import BufferMovieSim, CachedPSF

from simulator.background import BUFFER_MOVIES, gen_random_mov_stack
from simulator.constants import DEFAULT_NAVG, OPTIMUM_EVENT_DENSITY, RATIOMETRIC_RESCALE
from simulator.events import gen_doped_events
from simulator.ground_truth import gen_ground_truth
from simulator.ratiometric import gen_ratiometric_movie

PSF_PATH = Path(__file__).parent / "011_20241115_s_elo_wt_his_tag_7500x_expPSF.pickle"
_raw_psf = utils.load_from_pickle(str(PSF_PATH))


def exp_psf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate the experimental PSF at arbitrary (x, y) coordinates.

    Thin wrapper around the raw RectBivariateSpline interpolator that fixes
    `grid=False` so callers can pass flat coordinate arrays directly.

    Args:
        x: Coordinate array, in px, as returned by
            `CachedPSF._gen_offset_mgrid`.
        y: Coordinate array, in px, as returned by
            `CachedPSF._gen_offset_mgrid`.

    Returns:
        PSF amplitude at each (x, y) coordinate pair.
    """
    return _raw_psf(x, y, grid=False)


psf_model = CachedPSF(exp_psf, l_thum=21)
movie_simulator = BufferMovieSim(psf_model=psf_model)

calib = BUFFER_MOVIES[0].calibration


def c2m(contrast: np.ndarray) -> np.ndarray:
    """Convert contrast to mass, in kDa, via the primary buffer movie's calibration.

    Args:
        contrast: Contrast value(s).

    Returns:
        Mass, in kDa.
    """
    return np.divide(np.subtract(contrast, calib["intercept"]), calib["c2kDa"])


def m2c(mass: np.ndarray) -> np.ndarray:
    """Convert mass, in kDa, to contrast, via the primary buffer movie's calibration.

    Args:
        mass: Mass, in kDa.

    Returns:
        Contrast value(s).
    """
    return np.add(np.multiply(mass, calib["c2kDa"]), calib["intercept"])


event_generator = partial(
    gen_doped_events,
    # sorted since a negative calibration slope (as here) makes m2c decreasing
    # in mass, and Generator.uniform (unlike legacy np.random.uniform) raises
    # if low > high
    contrast_range=tuple(sorted((m2c(2000), m2c(6000)))),
    dopant_contrast_range=tuple(sorted((m2c(30), m2c(2000)))),
)


def gen_data(
    batch_size: int,
    length: int = 500,
    mov_thumbnail_size: int = 64,
    event_density_range: tuple[float, float] = (
        OPTIMUM_EVENT_DENSITY / 10,
        5 * OPTIMUM_EVENT_DENSITY,
    ),
    navg: int = DEFAULT_NAVG,
    ratiometric_rescale: float = RATIOMETRIC_RESCALE,
    seed: int | np.random.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Generate a batch of training-ready ratiometric movies and ground truth.

    For each sample: crops a random real background window
    (`gen_random_mov_stack`), samples events at a random density
    (`event_generator`), composites them onto the background
    (`movie_simulator.work`), converts to ratiometric contrast
    (`gen_ratiometric_movie`), and renders the matching heatmap/offset/
    orientation ground truth (`gen_ground_truth`). The `navg`-frame dead
    zone at each end (no valid ratiometric value, and where `gen_events`
    never places an event) is cropped from both the movie and the ground
    truth before batching, so nothing here needs NaN-handling downstream.

    Args:
        batch_size: Number of samples to generate.
        length: Number of frames in each sampled background window.
        mov_thumbnail_size: Width and height, in px, of each sampled
            background window.
        event_density_range: (low, high) event density, in events / um^2 /
            s, uniformly sampled per movie.
        navg: Ratiometric window size; also the number of frames excluded
            from event placement at each end of the movie (see `gen_events`).
        ratiometric_rescale: Scalar the ratiometric movie is multiplied by
            before batching (see `simulator.constants.RATIOMETRIC_RESCALE`).
        seed: Seed (or an existing `np.random.Generator`, reused as-is) for
            all randomness in this call -- the same seed reproduces the same
            batch. Defaults to fresh, unseeded randomness if not given.

    Returns:
        A tuple of:
            - Ratiometric movies, shape (batch_size, 1, T - 2*navg,
              mov_thumbnail_size, mov_thumbnail_size).
            - Ground truth dict with keys "heatmap" (batch_size, 3,
              T - 2*navg, H, W), "offset" and "orientation" (batch_size, 2,
              T - 2*navg, H, W) -- matching `EventDetector.forward`'s
              `predictions` up to the batch dimension.
    """
    rng = np.random.default_rng(seed)

    movies = []
    heatmaps = []
    offsets = []
    orientations = []

    for _ in range(batch_size):
        event_density = rng.uniform(*event_density_range)
        mov_stack = gen_random_mov_stack(
            length=length, mov_thumbnail_size=mov_thumbnail_size, rng=rng
        )

        events = event_generator(
            event_density=event_density, mov_shape=mov_stack.shape, navg=navg, rng=rng
        )
        simple_events = [record for event in events for record in event.to_simple()]

        ground_truth = gen_ground_truth(events=events, mov_shape=mov_stack.shape)
        sim_movie = movie_simulator.work(movie=mov_stack, events=simple_events)
        ratiometric = gen_ratiometric_movie(sim_movie, navg=navg) * ratiometric_rescale
        ratiometric = ratiometric[navg:-navg]  # drop the NaN dead zone

        movies.append(torch.from_numpy(ratiometric).unsqueeze(0))  # (1, T', H, W)
        heatmaps.append(ground_truth["heatmap"][:, navg:-navg])
        offsets.append(ground_truth["offset"][:, navg:-navg])
        orientations.append(ground_truth["orientation"][:, navg:-navg])

    return (
        torch.stack(movies),
        {
            "heatmap": torch.stack(heatmaps),
            "offset": torch.stack(offsets),
            "orientation": torch.stack(orientations),
        },
    )
