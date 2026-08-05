"""Shared physical, spatial, and calibration constants for the event simulator."""

BORDER_MASK: int = 5  # px excluded from each edge when placing events, keeping the full PSF footprint in-frame
NM_PER_PX: float = 72.6  # spatial calibration (nm/pixel)
FRAMES_PER_SECOND: float = 42.7  # acquisition frame rate (Hz)
HEATMAP_GAUSSIAN_SIGMA_PX: float = 1.0  # stdev of the ground truth heatmap gaussian, in px
HEATMAP_GAUSSIAN_THUMBNAIL_SIZE: int = 9  # px window the gaussian is truncated to (must fit within BORDER_MASK)
RATIOMETRIC_RESCALE: float = 2000.0  # approximates dividing by the empirical ratiometric noise stdev (sigma_noise)
OPTIMUM_EVENT_DENSITY: float = 0.5  # events / um^2 / s, the center of gen_data's default sampling range
DEFAULT_NAVG: int = 5  # default ratiometric window size (see ratiometric.gen_ratiometric_movie)
DEFAULT_MOV_THUMBNAIL_SIZE: int = 64  # px, training background window width & height
VAL_MOV_THUMBNAIL_SIZE: int = 48  # px, validation background window width & height (smaller-FoV buffer movies)
