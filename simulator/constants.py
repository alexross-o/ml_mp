"""Shared physical, spatial, and calibration constants for the event simulator."""

BORDER_MASK: int = 5  # px excluded from each edge when placing events, keeping the full PSF footprint in-frame
NM_PER_PX: float = 72.6  # spatial calibration (nm/pixel)
FRAMES_PER_SECOND: float = 42.7  # acquisition frame rate (Hz)
HEATMAP_GAUSSIAN_SIGMA_PX: float = 1.0  # stdev of the ground truth heatmap gaussian, in px
HEATMAP_GAUSSIAN_THUMBNAIL_SIZE: int = 9  # px window the gaussian is truncated to (must fit within BORDER_MASK)
RATIOMETRIC_RESCALE: float = 2000.0  # approximates dividing by the empirical ratiometric noise stdev (sigma_noise)
