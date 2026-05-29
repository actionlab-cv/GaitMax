"""GCaption attribute schema (GaitMax Table 1) — schema-first.

Seven attributes in a fixed order, so the cpt tensor's 7-attribute axis has stable semantics.
All constraints live here (single source of truth): multi-choice attributes are Enums, viewpoint is a nested enum object, and free-text attributes carry their format in the field description.
The prompt only frames the task.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

# fixed attribute order; embed/aggregate rely on this matching FrameCaption fields
ATTR_ORDER: tuple[str, ...] = (
    "age",
    "attire",
    "action",
    "related_item",
    "location",
    "viewpoint",
    "lighting",
)

# attributes aggregated by majority vote (discrete) vs. by centroid-closest frame (free-text)
CATEGORICAL: frozenset[str] = frozenset({"age", "action", "lighting", "viewpoint"})


class Age(StrEnum):
    YOUTH = "youth"
    YOUNG_ADULT = "young adult"
    ADULT = "adult"
    MID_AGES = "mid-ages"
    SENIOR = "senior"


class Action(StrEnum):
    STAND = "stand"
    WALK = "walk"
    RUN = "run"


class Lighting(StrEnum):
    BRIGHT = "bright"
    DIM = "dim"
    SHADOWED = "shadowed"
    NATURAL = "natural light"
    ARTIFICIAL = "artificial light"


class Roll(StrEnum):
    LEVEL = "level"
    TILTED = "tilted"


class Pitch(StrEnum):
    OVERHEAD = "overhead"
    EYE_LEVEL = "eye-level"
    LOW_ANGLE = "low-angle"


class Yaw(StrEnum):
    FRONT = "front"
    SIDE = "side"
    BACK = "back"


class Viewpoint(BaseModel):
    roll: Roll = Field(description="camera roll relative to the person")
    pitch: Pitch = Field(
        description="camera height: looking down (overhead), level (eye-level), or up (low-angle)"
    )
    yaw: Yaw = Field(description="which side of the person faces the camera")


class FrameCaption(BaseModel):
    age: Age
    attire: str = Field(
        description="[top color + top type] [bottom color + bottom type], e.g. 'white shirt + blue jeans'; one dominant color per garment"
    )
    action: Action
    related_item: list[str] = Field(
        default_factory=list,
        description="objects the person carries/interacts with, each '[adjective] [noun]'; [] if none",
    )
    location: str = Field(
        description="short scene/background description, e.g. 'an indoor corridor', 'a paved outdoor area'"
    )
    viewpoint: Viewpoint
    lighting: Lighting


class SequenceCaption(BaseModel):
    frames: list[FrameCaption]
