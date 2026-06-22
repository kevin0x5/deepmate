"""Built-in code-native pixel pet frame packs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PixelPet:
    """A small code-native pixel pet pack."""

    pet_id: str
    species: str
    style: str
    palette: dict[str, str]
    frames: dict[str, tuple[tuple[str, ...], ...]]
    fps: dict[str, int]


BASE_DOG = (
    "....................",
    "...BBB........BBB...",
    "..BCCCBB....BBCCCB..",
    ".BCCCCCCBBBBCCCCCCB.",
    ".BCCCCCCCCCCCCCCCCB.",
    ".BCCWWCCCCCCCCWWCCB.",
    ".BCCBBCCCPCCBBCCCB..",
    "..BCCCCCCCCCCCCCCB..",
    "...BCCCGGGGCCCB.....",
    "..BBCCCCCCCCCCBB....",
    ".B..BCCCCCCCCB..B...",
    "....BCCCCCCCCB......",
    "....BB......BB......",
    "...B..B....B..B.....",
    "....................",
    "....................",
)

BASE_CAT = (
    "....................",
    "..BB..........BB....",
    ".BSSB........BSSB...",
    ".BCCCBBBBBBBBCCCB...",
    ".BCCCCCCCCCCCCCCB...",
    ".BCCWWCCCCCCWWCCB...",
    ".BCCBBCPCCCBCCCB....",
    "B..BCCCCCCCCCCB..B..",
    "...BCCCGGGGCCCB.....",
    "..BBCCCCCCCCCCBB....",
    "....BCCCCCCCCB......",
    "....BCCCCCCCCB......",
    "....BB......BB......",
    "...B..B....B..B.....",
    "....................",
    "....................",
)

BASE_SQUIRREL = (
    "....................",
    "...........BBBB.....",
    "..........BCCCCB....",
    ".....BBBBBCCCCCB....",
    "....BCCCCCCCCCCB....",
    "...BCCCWWCCCCWWB....",
    "...BCCBBCPCCBB......",
    "....BCCCGGGCB.......",
    "...BBCCCCCCCCB......",
    "..B..BCCCCCCCB......",
    ".....BCCCCCCCB......",
    ".....BB....BBB......",
    "....B..B..B..B......",
    "....................",
    "....................",
    "....................",
)

BASE_PENGUIN = (
    "....................",
    "......BBBBBBBB......",
    ".....BSSSSSSSSB.....",
    "....BSSCCCCCCSSB....",
    "....BSCWWCCWWCSB....",
    "....BSCBBCCBBCSB....",
    "....BSCCCPCCCCSB....",
    ".....BSGGGGCCSB.....",
    "......BCCCCCCB......",
    ".....BBCCCCCCBB.....",
    "....B..BCCCCB..B....",
    ".......BG..GB.......",
    "......G......G......",
    "....................",
    "....................",
    "....................",
)


def built_in_pet(pet_id: str) -> PixelPet:
    """Return a built-in pet pack by id, falling back to dog."""
    return BUILT_IN_PETS.get(pet_id, BUILT_IN_PETS["dog-happy"])


def built_in_pet_ids() -> tuple[str, ...]:
    """Return stable built-in pet ids."""
    return tuple(BUILT_IN_PETS)


def _states(base: tuple[str, ...]) -> dict[str, tuple[tuple[str, ...], ...]]:
    return {
        "idle": (base, _shift(base, 0, 1)),
        "thinking": (base, _mark(base, "G"), _shift(_mark(base, "G"), 1, 0)),
        "working": (base, _shift(base, 1, 0), _shift(base, -1, 0)),
        "waiting": (_mark(base, "Y"), _shift(_mark(base, "Y"), 0, -1)),
        "reporting": (_mark(base, "L"), _shift(_mark(base, "L"), 0, -1)),
        "celebrate": (_mark(base, "G"), _shift(_mark(base, "G"), 0, -1)),
        "blocked": (_mark(base, "R"), _shift(_mark(base, "R"), 0, 1)),
        "resting": (_shift(base, 0, 1), _shift(base, 0, 2)),
        "offline": (_shift(base, 0, 2),),
    }


def _shift(frame: tuple[str, ...], x: int, y: int) -> tuple[str, ...]:
    rows = list(frame)
    if y > 0:
        rows = ["." * len(rows[0])] * y + rows[:-y]
    elif y < 0:
        rows = rows[-y:] + ["." * len(rows[0])] * (-y)
    shifted: list[str] = []
    for row in rows:
        if x > 0:
            shifted.append("." * x + row[:-x])
        elif x < 0:
            shifted.append(row[-x:] + "." * (-x))
        else:
            shifted.append(row)
    return tuple(shifted)


def _mark(frame: tuple[str, ...], code: str) -> tuple[str, ...]:
    rows = list(frame)
    if not rows:
        return frame
    row = rows[0]
    rows[0] = row[:-1] + code if row else code
    return tuple(rows)


BUILT_IN_PETS: dict[str, PixelPet] = {
    "dog-happy": PixelPet(
        pet_id="dog-happy",
        species="dog",
        style="happy",
        palette={
            ".": "",
            "B": "#3d2b1f",
            "C": "#c98245",
            "W": "#fff4df",
            "P": "#f58ca8",
            "G": "#ffd166",
            "Y": "#f4c430",
            "R": "#e85d75",
            "L": "#7cc7ff",
            "S": "#fff4df",
        },
        frames=_states(BASE_DOG),
        fps={},
    ),
    "cat-lazy": PixelPet(
        pet_id="cat-lazy",
        species="cat",
        style="lazy",
        palette={
            ".": "",
            "B": "#3f4657",
            "C": "#8fa3bf",
            "W": "#f5f7fb",
            "P": "#b7a6d9",
            "G": "#f3f0ff",
            "Y": "#d8c76f",
            "R": "#d7728a",
            "L": "#91b8ff",
            "S": "#d7deea",
        },
        frames=_states(BASE_CAT),
        fps={},
    ),
    "squirrel-lively": PixelPet(
        pet_id="squirrel-lively",
        species="squirrel",
        style="lively",
        palette={
            ".": "",
            "B": "#4a2d1f",
            "C": "#9a5b2f",
            "W": "#ffe6bd",
            "P": "#dd8f45",
            "G": "#8ab17d",
            "Y": "#f2c14e",
            "R": "#dd6b6b",
            "L": "#77c7c2",
            "S": "#6d8f57",
        },
        frames=_states(BASE_SQUIRREL),
        fps={},
    ),
    "penguin-naive": PixelPet(
        pet_id="penguin-naive",
        species="penguin",
        style="naive",
        palette={
            ".": "",
            "B": "#1d2633",
            "C": "#f7fbff",
            "W": "#98d8ef",
            "P": "#ffb4a2",
            "G": "#f4c430",
            "Y": "#f4c430",
            "R": "#e76f8a",
            "L": "#80d8ff",
            "S": "#bdefff",
        },
        frames=_states(BASE_PENGUIN),
        fps={},
    ),
}
