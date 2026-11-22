"""Built-in pixel pet packs and their compact frame representation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True, slots=True)
class PixelPet:
    """A small code-native pixel pet pack."""

    pet_id: str
    species: str
    style: str
    palette: dict[str, str]
    frames: dict[str, tuple[tuple[str, ...], ...]]
    fps: dict[str, int]


def _compact_frame(rows: list[str]) -> tuple[str, ...]:
    """Sample the shared square artwork into the original 20 × 16 frame."""
    size = len(rows)
    centers = [(2 * i + 1) * size // 32 for i in range(16)]
    return tuple(".." + "".join(rows[y][x] for x in centers) + ".." for y in centers)


def _shift(frame: tuple[str, ...], dx: int, dy: int) -> tuple[str, ...]:
    blank = "." * len(frame[0])
    rows = frame
    if dy > 0:
        rows = (blank,) * dy + frame[:-dy]
    elif dy < 0:
        rows = frame[-dy:] + (blank,) * -dy
    if dx > 0:
        return tuple("." * dx + row[:-dx] for row in rows)
    if dx < 0:
        return tuple(row[-dx:] + "." * -dx for row in rows)
    return rows


def _make_pack(pet_id: str, portrait: dict) -> PixelPet:
    species, style = pet_id.split("-", 1)
    open_rows = portrait["rows"]
    closed_rows = open_rows.copy()
    for patch in portrait["closedEyes"]:
        left = patch["x"]
        for y, pixels in enumerate(patch["rows"], patch["y"]):
            row = closed_rows[y]
            closed_rows[y] = row[:left] + pixels + row[left + len(pixels):]
    awake = _compact_frame(open_rows)
    resting = _compact_frame(closed_rows)
    up = _shift(awake, 0, -1)
    down = _shift(awake, 0, 1)
    return PixelPet(
        pet_id=pet_id,
        species=species,
        style=style,
        palette={".": "", **portrait["palette"]},
        frames={
            "idle": (awake, awake, resting, awake),
            "thinking": (awake, _shift(awake, 1, 0)),
            "working": (awake, up, awake, down),
            "waiting": (awake, up),
            "reporting": (awake, _shift(awake, -1, 0)),
            "celebrate": (up, awake, _shift(awake, 1, -1)),
            "blocked": (awake, down),
            "resting": (resting, _shift(resting, 0, 1)),
            "offline": (resting,),
        },
        fps={},
    )


def _load_packs() -> dict[str, PixelPet]:
    resource = files("deepmate").joinpath("pet_ui", "renderer", "portraits.json")
    portraits = json.loads(resource.read_text(encoding="utf-8"))
    return {
        pet_id: _make_pack(pet_id, portraits[pet_id.split("-", 1)[0]])
        for pet_id in ("dog-happy", "cat-lazy", "squirrel-lively", "penguin-naive")
    }


BUILT_IN_PETS = _load_packs()
BASE_DOG = BUILT_IN_PETS["dog-happy"].frames["idle"][0]
BASE_CAT = BUILT_IN_PETS["cat-lazy"].frames["idle"][0]
BASE_SQUIRREL = BUILT_IN_PETS["squirrel-lively"].frames["idle"][0]
BASE_PENGUIN = BUILT_IN_PETS["penguin-naive"].frames["idle"][0]


def built_in_pet(pet_id: str) -> PixelPet:
    """Return a built-in pet pack by id, falling back to dog."""
    return BUILT_IN_PETS.get(pet_id, BUILT_IN_PETS["dog-happy"])


def built_in_pet_ids() -> tuple[str, ...]:
    """Return stable built-in pet ids."""
    return tuple(BUILT_IN_PETS)
