"""Simulated binding, unbinding, and movement (dipole) events."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from simulator.constants import BORDER_MASK, FRAMES_PER_SECOND, NM_PER_PX

# Movement events are sampled in three distance tiers rather than uniformly
# across `distance_range`, biasing toward the short, often sub-pixel hops
# real single-molecule movements tend to produce.
_MOVEMENT_SMALL_DISTANCE_FRACTION: float = 0.6
_MOVEMENT_MEDIUM_DISTANCE_FRACTION: float = 0.3
_MOVEMENT_SMALL_DISTANCE_MAX_NM: float = 20.0
_MOVEMENT_MEDIUM_DISTANCE_MAX_NM: float = 100.0


def density_to_n_events(
    density: float,
    shape: tuple[int, int, int],
) -> int:
    """Convert an event density to an expected event count for a movie.

    Args:
        density: Event density, in events / um^2 / s.
        shape: Movie array shape (T, H, W).

    Returns:
        Expected number of events over the movie's full duration and area.
    """
    area_um2 = (shape[1] * NM_PER_PX / 1000) * (shape[2] * NM_PER_PX / 1000)
    duration_s = shape[0] / FRAMES_PER_SECOND

    return int(density * area_um2 * duration_s)


class AbstractSimEvent(ABC):
    """Interface for a single simulated event placed into a training movie.

    Concrete events store their (x, y, i, c) coordinates as plain fields;
    this interface only constrains the values derived from them.
    """

    x: float
    y: float
    i: float
    c: float

    @property
    @abstractmethod
    def hot_px(self) -> tuple[int, int]:
        """Nearest integer (x, y) pixel, i.e. the heatmap peak location."""

    @property
    @abstractmethod
    def offset(self) -> tuple[float, float]:
        """Sub-pixel (dx, dy) offset of (x, y) from `hot_px`."""

    @abstractmethod
    def to_simple(self) -> list[list[float]]:
        """Flatten the event to one or more [x, y, i, c] records."""


@dataclass
class BaseSimEvent(AbstractSimEvent):
    """Concrete single-point event: a binding or unbinding at (x, y, i, c).

    Attributes:
        x: Sub-pixel x-coordinate.
        y: Sub-pixel y-coordinate.
        i: Frame index (temporal coordinate).
        c: Contrast.
    """

    x: float
    y: float
    i: float
    c: float

    @property
    def hot_px(self) -> tuple[int, int]:
        return (np.round(self.x).astype(int), np.round(self.y).astype(int))

    @property
    def offset(self) -> tuple[float, float]:
        x_px, y_px = self.hot_px

        return (self.x - float(x_px), self.y - float(y_px))

    def to_simple(self) -> list[list[float]]:
        return [[self.x, self.y, self.i, self.c]]


class BindingSimEvent(BaseSimEvent):
    """A particle landing (binding) event."""


class UnbindingSimEvent(BaseSimEvent):
    """A particle leaving (unbinding) event."""


@dataclass
class MovementSimEvent(BaseSimEvent):
    """A movement: a linked unbinding-then-binding pair.

    Represents a particle moving from one location to another, modeled as an
    unbinding event and a binding event straddling the midpoint (x, y),
    separated by `distance` at angle `theta`. Both endpoints share the same
    intensity/frame index and contrast.

    Attributes:
        distance: Distance between the unbinding and binding endpoints.
        theta: Direction of travel, in radians.
    """

    distance: float
    theta: float

    def __post_init__(self) -> None:
        self.theta = (self.theta / (2 * np.pi)) - np.floor(
            self.theta / (2 * np.pi)
        )  # wrap to [0, 2*pi)

        self.dx: float = self.distance * np.cos(self.theta)
        self.dy: float = self.distance * np.sin(self.theta)

        self.unbinding: UnbindingSimEvent = UnbindingSimEvent(
            x=self.x - self.dx / 2, y=self.y - self.dy / 2, i=self.i, c=-self.c
        )
        self.binding: BindingSimEvent = BindingSimEvent(
            x=self.x + self.dx / 2, y=self.y + self.dy / 2, i=self.i, c=self.c
        )

    def to_simple(self) -> list[list[float]]:
        """Flatten to the unbinding and binding endpoint records.

        Returns:
            A list of two [x, y, i, c] records: the unbinding endpoint
            followed by the binding endpoint.
        """
        return self.unbinding.to_simple() + self.binding.to_simple()


def gen_events(
    event_density: float,
    mov_shape: tuple[int, int, int],
    contrast_range: tuple[float, float],
    distance_range: tuple[float, float] = (1.5, 300),
    event_type_weight: tuple[float, float, float] = (1, 1, 1),
    navg: int = 5,
    rng: np.random.Generator | None = None,
) -> Sequence[AbstractSimEvent]:
    """Sample a batch of random binding, unbinding, and movement events.

    Args:
        event_density: Target combined event density, in events / um^2 / s,
            split across the three event types by `event_type_weight`.
        mov_shape: Movie array shape (T, H, W) events are placed within.
        contrast_range: (low, high) contrast sampled for each event's
            arrival endpoint; departures use the negated contrast.
        distance_range: (low, high) nm distance sampled for each movement
            event's unbinding-binding separation (converted to px via
            `NM_PER_PX`).
        event_type_weight: Relative weighting of (binding, unbinding,
            movement) events; need not sum to 1.
        navg: Number of frames averaged into each side of a ratiometric
            window (see `simulator.ratiometric.gen_ratiometric_movie`).
            Events are never placed in the first/last `navg` frames, since
            those have no valid ratiometric value -- avoiding any need to
            handle them downstream.
        rng: Random generator to sample from. Defaults to a fresh, unseeded
            `np.random.default_rng()` if not given.

    Returns:
        The generated events, in no particular order.

    Raises:
        ValueError: If `event_type_weight` doesn't have exactly 3 elements,
            if `mov_shape`'s time axis is too short to leave any frames
            outside the `navg` exclusion zone, or if its spatial axes are
            too short to leave any placement area outside `BORDER_MASK`.
    """
    if len(event_type_weight) != 3:
        raise ValueError("length of weights must be equal to no. of event types")
    rng = np.random.default_rng(rng)
    if mov_shape[0] <= 2 * navg:
        raise ValueError(
            f"mov_shape[0] ({mov_shape[0]}) must exceed 2 * navg ({2 * navg}) "
            "to leave any frames outside the ratiometric exclusion zone"
        )
    if mov_shape[1] <= 2 * BORDER_MASK or mov_shape[2] <= 2 * BORDER_MASK:
        raise ValueError(
            f"mov_shape[1:] {mov_shape[1:]} must exceed 2 * BORDER_MASK "
            f"({2 * BORDER_MASK}) in each spatial dimension"
        )

    weights = np.array(event_type_weight) / np.sum(event_type_weight)

    mov_shape_masked = (
        mov_shape[0],
        mov_shape[1] - 2 * BORDER_MASK,
        mov_shape[2] - 2 * BORDER_MASK,
    )

    n_bindings, n_unbindings, n_movements = (
        density_to_n_events(density=density, shape=mov_shape_masked)
        for density in weights * event_density
    )

    # half-pixel margin inside BORDER_MASK so a rounded hot_px never falls in the masked border
    x_low, x_high = BORDER_MASK - 0.5, mov_shape[2] - (BORDER_MASK - 0.5)
    y_low, y_high = BORDER_MASK - 0.5, mov_shape[1] - (BORDER_MASK - 0.5)

    # exclude the first/last navg frames, which have no valid ratiometric value
    i_low, i_high = navg, mov_shape[0] - navg

    binding_evs = [
        BindingSimEvent(x=x, y=y, i=i, c=c)
        for x, y, i, c in zip(
            rng.uniform(x_low, x_high, n_bindings),
            rng.uniform(y_low, y_high, n_bindings),
            rng.uniform(i_low, i_high, n_bindings),
            rng.uniform(contrast_range[0], contrast_range[1], n_bindings),
        )
    ]
    unbinding_evs = [
        UnbindingSimEvent(x=x, y=y, i=i, c=c)
        for x, y, i, c in zip(
            rng.uniform(x_low, x_high, n_unbindings),
            rng.uniform(y_low, y_high, n_unbindings),
            rng.uniform(i_low, i_high, n_unbindings),
            -rng.uniform(contrast_range[0], contrast_range[1], n_unbindings),
        )
    ]

    n_small_movements = int(_MOVEMENT_SMALL_DISTANCE_FRACTION * n_movements)
    n_medium_movements = int(_MOVEMENT_MEDIUM_DISTANCE_FRACTION * n_movements)
    n_large_movements = n_movements - n_small_movements - n_medium_movements

    movement_evs = [
        MovementSimEvent(x=x, y=y, i=i, c=c, distance=distance / NM_PER_PX, theta=theta)
        for x, y, i, c, distance, theta in zip(
            rng.uniform(x_low, x_high, n_movements),
            rng.uniform(y_low, y_high, n_movements),
            rng.uniform(i_low, i_high, n_movements),
            rng.uniform(contrast_range[0], contrast_range[1], n_movements),
            np.concatenate(
                [
                    rng.uniform(
                        distance_range[0], _MOVEMENT_SMALL_DISTANCE_MAX_NM, n_small_movements
                    ),
                    rng.uniform(
                        _MOVEMENT_SMALL_DISTANCE_MAX_NM,
                        _MOVEMENT_MEDIUM_DISTANCE_MAX_NM,
                        n_medium_movements,
                    ),
                    rng.uniform(
                        _MOVEMENT_MEDIUM_DISTANCE_MAX_NM, distance_range[1], n_large_movements
                    ),
                ]
            ),
            rng.uniform(0, 2 * np.pi, n_movements),
        )
    ]

    return binding_evs + unbinding_evs + movement_evs


def gen_doped_events(
    event_density: float,
    mov_shape: tuple[int, int, int],
    contrast_range: tuple[float, float],
    dopant_contrast_range: tuple[float, float],
    distance_range: tuple[float, float] = (1.5, 300),
    event_type_weight: tuple[float, float, float] = (1, 1, 1),
    dopant_density_fraction: float = 0.2,
    navg: int = 5,
    rng: np.random.Generator | None = None,
) -> Sequence[AbstractSimEvent]:
    """Wraps `gen_events`, adding binding/unbinding-only "dopant" events at
    `dopant_contrast_range` so the model also sees events outside the
    primary population's mass range.

    Args:
        dopant_contrast_range: (low, high) contrast for the dopant events.
        dopant_density_fraction: Dopant density as a fraction of
            `event_density`, split evenly between binding and unbinding.
        See `gen_events` for the remaining args.

    Returns:
        The primary and dopant events combined, in no particular order.
    """
    rng = np.random.default_rng(rng)
    events = gen_events(
        event_density=event_density,
        mov_shape=mov_shape,
        contrast_range=contrast_range,
        distance_range=distance_range,
        event_type_weight=event_type_weight,
        navg=navg,
        rng=rng,
    )
    dopant_events = gen_events(
        event_density=dopant_density_fraction * event_density,
        mov_shape=mov_shape,
        contrast_range=dopant_contrast_range,
        distance_range=distance_range,
        event_type_weight=(1, 1, 0),
        navg=navg,
        rng=rng,
    )

    return events + dopant_events
