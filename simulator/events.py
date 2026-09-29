"""A registry of simulated event types (binding, unbinding, movement, ...).

New event types are added by subclassing `AbstractSimEvent` (or the
single-point `BaseSimEvent`) and decorating the class with
`@register_event_type`, which appends it to `EVENT_TYPES` and stamps its
`class_index`. Everything downstream (heatmap channel count, per-class
sampling weights, orientation-loss masking) derives from `EVENT_TYPES`
rather than hardcoding the current three types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Mapping, Sequence, TypeVar

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


_T = TypeVar("_T", bound="AbstractSimEvent")


class AbstractSimEvent(ABC):
    """Interface for a registered, pluggable simulated event type.

    Concrete events store their (x, y, i, c) coordinates as plain fields;
    this interface constrains the values derived from them, plus the two
    hooks (`sample`, and the `name`/`has_orientation`/`class_index` class
    attributes) that let `gen_events` and `gen_ground_truth` handle any
    registered type polymorphically, with no per-type special-casing
    outside the type's own class body.
    """

    name: ClassVar[str]
    has_orientation: ClassVar[bool]
    class_index: ClassVar[int]  # stamped by register_event_type, not hand-written

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

    @property
    @abstractmethod
    def orientation(self) -> tuple[float, float] | None:
        """(cos, sin) unit orientation vector, or None if this event type
        has no orientation (`has_orientation = False`)."""

    @abstractmethod
    def to_simple(self) -> list[list[float]]:
        """Flatten the event to one or more [x, y, i, c] records."""

    @classmethod
    @abstractmethod
    def sample(
        cls: type[_T],
        n: int,
        x_low: float,
        x_high: float,
        y_low: float,
        y_high: float,
        i_low: float,
        i_high: float,
        contrast_range: tuple[float, float],
        rng: np.random.Generator,
        **type_kwargs,
    ) -> list[_T]:
        """Sample `n` random instances of this event type.

        Args:
            n: Number of events to sample.
            x_low, x_high: (x) placement bounds, in px.
            y_low, y_high: (y) placement bounds, in px.
            i_low, i_high: Frame-index placement bounds.
            contrast_range: (low, high) contrast sampled for each event.
            rng: Random generator to sample from.
            **type_kwargs: Extra shared context (e.g. `distance_range`)
                that only some event types use; others should ignore it.

        Returns:
            The sampled events, as concrete instances of this type.
        """


EVENT_TYPES: list[type[AbstractSimEvent]] = []


def register_event_type(cls: type[_T]) -> type[_T]:
    """Register a concrete `AbstractSimEvent` subclass in `EVENT_TYPES`,
    stamping its `class_index` to its position in the registry."""
    cls.class_index = len(EVENT_TYPES)
    EVENT_TYPES.append(cls)
    return cls


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

    @property
    def orientation(self) -> tuple[float, float] | None:
        return None

    def to_simple(self) -> list[list[float]]:
        return [[self.x, self.y, self.i, self.c]]

    @classmethod
    def _sample_point_event(
        cls: type[_T],
        n: int,
        x_low: float,
        x_high: float,
        y_low: float,
        y_high: float,
        i_low: float,
        i_high: float,
        contrast_range: tuple[float, float],
        rng: np.random.Generator,
        contrast_sign: float = 1.0,
        **_type_kwargs,
    ) -> list[_T]:
        """Shared sampler for single-point event types (binding/unbinding):
        uniform (x, y, i), contrast uniform in `contrast_range` and negated
        when `contrast_sign = -1.0`."""
        xs = rng.uniform(x_low, x_high, n)
        ys = rng.uniform(y_low, y_high, n)
        ivals = rng.uniform(i_low, i_high, n)
        cs = contrast_sign * rng.uniform(contrast_range[0], contrast_range[1], n)

        return [cls(x=x, y=y, i=i, c=c) for x, y, i, c in zip(xs, ys, ivals, cs)]


@register_event_type
class BindingSimEvent(BaseSimEvent):
    """A particle landing (binding) event."""

    name: ClassVar[str] = "binding"
    has_orientation: ClassVar[bool] = False

    @classmethod
    def sample(cls, n: int, **kwargs) -> list["BindingSimEvent"]:
        return cls._sample_point_event(n, contrast_sign=1.0, **kwargs)


@register_event_type
class UnbindingSimEvent(BaseSimEvent):
    """A particle leaving (unbinding) event."""

    name: ClassVar[str] = "unbinding"
    has_orientation: ClassVar[bool] = False

    @classmethod
    def sample(cls, n: int, **kwargs) -> list["UnbindingSimEvent"]:
        return cls._sample_point_event(n, contrast_sign=-1.0, **kwargs)


@register_event_type
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

    name: ClassVar[str] = "movement"
    has_orientation: ClassVar[bool] = True

    distance: float
    theta: float

    def __post_init__(self) -> None:
        self.theta = np.mod(self.theta, 2 * np.pi)  # wrap to [0, 2*pi)

        self.dx: float = self.distance * np.cos(self.theta)
        self.dy: float = self.distance * np.sin(self.theta)

        self.unbinding: UnbindingSimEvent = UnbindingSimEvent(
            x=self.x - self.dx / 2, y=self.y - self.dy / 2, i=self.i, c=-self.c
        )
        self.binding: BindingSimEvent = BindingSimEvent(
            x=self.x + self.dx / 2, y=self.y + self.dy / 2, i=self.i, c=self.c
        )

    @property
    def orientation(self) -> tuple[float, float] | None:
        return (np.cos(self.theta), np.sin(self.theta))

    def to_simple(self) -> list[list[float]]:
        """Flatten to the unbinding and binding endpoint records.

        Returns:
            A list of two [x, y, i, c] records: the unbinding endpoint
            followed by the binding endpoint.
        """
        return self.unbinding.to_simple() + self.binding.to_simple()

    @classmethod
    def sample(
        cls,
        n: int,
        x_low: float,
        x_high: float,
        y_low: float,
        y_high: float,
        i_low: float,
        i_high: float,
        contrast_range: tuple[float, float],
        rng: np.random.Generator,
        distance_range: tuple[float, float] = (1.5, 300),
        **_type_kwargs,
    ) -> list["MovementSimEvent"]:
        n_small = int(_MOVEMENT_SMALL_DISTANCE_FRACTION * n)
        n_medium = int(_MOVEMENT_MEDIUM_DISTANCE_FRACTION * n)
        n_large = n - n_small - n_medium

        xs = rng.uniform(x_low, x_high, n)
        ys = rng.uniform(y_low, y_high, n)
        ivals = rng.uniform(i_low, i_high, n)
        cs = rng.uniform(contrast_range[0], contrast_range[1], n)
        distances = np.concatenate(
            [
                rng.uniform(distance_range[0], _MOVEMENT_SMALL_DISTANCE_MAX_NM, n_small),
                rng.uniform(
                    _MOVEMENT_SMALL_DISTANCE_MAX_NM,
                    _MOVEMENT_MEDIUM_DISTANCE_MAX_NM,
                    n_medium,
                ),
                rng.uniform(_MOVEMENT_MEDIUM_DISTANCE_MAX_NM, distance_range[1], n_large),
            ]
        )
        thetas = rng.uniform(0, 2 * np.pi, n)

        return [
            cls(x=x, y=y, i=i, c=c, distance=distance / NM_PER_PX, theta=theta)
            for x, y, i, c, distance, theta in zip(xs, ys, ivals, cs, distances, thetas)
        ]


def gen_events(
    event_density: float,
    mov_shape: tuple[int, int, int],
    contrast_range: tuple[float, float],
    distance_range: tuple[float, float] = (1.5, 300),
    event_type_weight: Mapping[str, float] | None = None,
    navg: int = 5,
    rng: np.random.Generator | None = None,
) -> Sequence[AbstractSimEvent]:
    """Sample a batch of random events, one `sample()` call per `EVENT_TYPES` entry.

    Args:
        event_density: Target combined event density, in events / um^2 / s,
            split across event types by `event_type_weight`.
        mov_shape: Movie array shape (T, H, W) events are placed within.
        contrast_range: (low, high) contrast sampled for each event's
            arrival endpoint; departures use the negated contrast.
        distance_range: (low, high) nm distance sampled for each movement
            event's unbinding-binding separation (converted to px via
            `NM_PER_PX`). Ignored by event types that don't use it.
        event_type_weight: Relative weighting of event types, keyed by each
            registered type's `name` (see `EVENT_TYPES`); need not sum to 1.
            Defaults to equal weight across all registered types.
        navg: Number of frames averaged into each side of a ratiometric
            window (see `simulator.ratiometric.gen_ratiometric_movie`).
            Events are never placed in the first/last `navg` frames, since
            those have no valid ratiometric value -- avoiding any need to
            handle them downstream.
        rng: Random generator to sample from. Defaults to a fresh, unseeded
            `np.random.default_rng()` if not given.

    Returns:
        The generated events, grouped by `EVENT_TYPES` order (all of one
        type, then all of the next, ...).

    Raises:
        ValueError: If `event_type_weight` has a key that isn't a registered
            event type's `name`, if `mov_shape`'s time axis is too short to
            leave any frames outside the `navg` exclusion zone, or if its
            spatial axes are too short to leave any placement area outside
            `BORDER_MASK`.
    """
    if event_type_weight is None:
        event_type_weight = {event_type.name: 1.0 for event_type in EVENT_TYPES}
    known_names = {event_type.name for event_type in EVENT_TYPES}
    unrecognized = set(event_type_weight) - known_names
    if unrecognized:
        raise ValueError(f"unrecognized event type(s) in event_type_weight: {sorted(unrecognized)}")

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

    weights = np.array(
        [event_type_weight.get(event_type.name, 0.0) for event_type in EVENT_TYPES]
    )
    weights = weights / weights.sum()

    mov_shape_masked = (
        mov_shape[0],
        mov_shape[1] - 2 * BORDER_MASK,
        mov_shape[2] - 2 * BORDER_MASK,
    )

    # half-pixel margin inside BORDER_MASK so a rounded hot_px never falls in the masked border
    x_low, x_high = BORDER_MASK - 0.5, mov_shape[2] - (BORDER_MASK - 0.5)
    y_low, y_high = BORDER_MASK - 0.5, mov_shape[1] - (BORDER_MASK - 0.5)

    # exclude the first/last navg frames, which have no valid ratiometric value
    i_low, i_high = navg, mov_shape[0] - navg

    events: list[AbstractSimEvent] = []
    for event_type, weight in zip(EVENT_TYPES, weights):
        n = density_to_n_events(density=weight * event_density, shape=mov_shape_masked)
        events.extend(
            event_type.sample(
                n,
                x_low=x_low,
                x_high=x_high,
                y_low=y_low,
                y_high=y_high,
                i_low=i_low,
                i_high=i_high,
                contrast_range=contrast_range,
                rng=rng,
                distance_range=distance_range,
            )
        )

    return events


def gen_doped_events(
    event_density: float,
    mov_shape: tuple[int, int, int],
    contrast_range: tuple[float, float],
    dopant_contrast_range: tuple[float, float],
    distance_range: tuple[float, float] = (1.5, 300),
    event_type_weight: Mapping[str, float] | None = None,
    dopant_density_fraction: float = 0.2,
    navg: int = 5,
    rng: np.random.Generator | None = None,
) -> Sequence[AbstractSimEvent]:
    """Wraps `gen_events`, adding non-oriented-only "dopant" events at
    `dopant_contrast_range` so the model also sees events outside the
    primary population's mass range.

    Args:
        dopant_contrast_range: (low, high) contrast for the dopant events.
        dopant_density_fraction: Dopant density as a fraction of
            `event_density`, split evenly across event types with
            `has_orientation = False` (e.g. binding/unbinding; oriented
            types like movement are excluded from doping).
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
    dopant_event_type_weight = {
        event_type.name: 0.0 if event_type.has_orientation else 1.0
        for event_type in EVENT_TYPES
    }
    dopant_events = gen_events(
        event_density=dopant_density_fraction * event_density,
        mov_shape=mov_shape,
        contrast_range=dopant_contrast_range,
        distance_range=distance_range,
        event_type_weight=dopant_event_type_weight,
        navg=navg,
        rng=rng,
    )

    return events + dopant_events
