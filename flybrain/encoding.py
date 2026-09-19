"""Turn characters into spike trains.

A 5x7 bitmap stands in for a 35-unit receptor sheet. Each "pixel" drives a
Poisson process whose rate depends on whether the pixel is ink or blank, which
is the standard rate-coding convention for feeding static patterns into an SNN.

Two vocabularies, and the distinction matters
---------------------------------------------
``FONT_5X7``   27 glyphs: space plus A-Z. **This is the fly's vocabulary.**
               ``CLASSES`` is built from it, so there are 27 MBON output cells
               and no more. The fly can only ever *say* one of these 27 things.

``FONT_DIGITS`` 10 more glyphs: 0-9. These are **input only.** There is no
               MBON for a digit and there never will be, so a digit can be
               shown to the receptors but can never be answered correctly -
               the readout is physically incapable of emitting one. Whatever
               the fly says in response to a digit is necessarily wrong, and
               that is a property of the architecture, not of training.

``INPUT_GLYPHS`` is the union, i.e. everything that can be rasterised onto the
receptor sheet. ``char_bitmap`` reads from it, so training (which only ever
visits ``CLASSES``) and digit presentation share one code path.
"""

from __future__ import annotations

from typing import Iterable

import torch

H, W = 7, 5
N_PIXELS = H * W

FONT_5X7: dict[str, list[str]] = {
    " ": [".....", ".....", ".....", ".....", ".....", ".....", "....."],
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "B": ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
    "C": [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
    "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "F": ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
    "G": [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "I": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"],
    "J": ["..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    "M": ["#...#", "##.##", "#.#.#", "#...#", "#...#", "#...#", "#...#"],
    "N": ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"],
    "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    "Q": [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "V": ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    "W": ["#...#", "#...#", "#...#", "#...#", "#.#.#", "##.##", "#...#"],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    "Y": ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    "Z": ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
}

# Digits. These are INPUT ONLY - see the module docstring. They are never added
# to CLASSES and the network is never trained on them, which is precisely what
# makes the digit demonstration meaningful: the bitmaps are real, the forward
# pass is real, and the answer is real, but the output layer has no digit to
# give back.
FONT_DIGITS: dict[str, list[str]] = {
    "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
}

# Everything that can be drawn on the receptor sheet, trained or not.
INPUT_GLYPHS: dict[str, list[str]] = {**FONT_5X7, **FONT_DIGITS}

# The fly's vocabulary: one MBON per entry. 27, and unchanged.
CLASSES: list[str] = sorted(FONT_5X7)
N_CLASSES = len(CLASSES)


def is_class(ch: str) -> bool:
    """True if the fly has an output cell for this character (i.e. it can say it)."""
    return ch.upper() in FONT_5X7


def is_input(ch: str) -> bool:
    """True if a 5x7 glyph exists for this character (i.e. it can be presented)."""
    return ch.upper() in INPUT_GLYPHS


def char_bitmap(ch: str) -> torch.Tensor:
    """Return the clean 35-element {0,1} bitmap for a presentable character."""
    ch = ch.upper()
    if ch not in INPUT_GLYPHS:
        raise KeyError(f"no glyph for {ch!r}")
    rows = INPUT_GLYPHS[ch]
    flat = [1.0 if c == "#" else 0.0 for row in rows for c in row]
    return torch.tensor(flat, dtype=torch.float32)


def render(ch: str) -> str:
    """ASCII preview, handy for debugging."""
    return "\n".join(INPUT_GLYPHS[ch.upper()])


def _augment(
    bitmap: torch.Tensor,
    gen: torch.Generator,
    noise_p: float,
    shift_prob: float,
) -> torch.Tensor:
    grid = bitmap.reshape(H, W)

    if shift_prob > 0 and float(torch.rand(1, generator=gen)) < shift_prob:
        dy = int(torch.randint(-1, 2, (1,), generator=gen))
        dx = int(torch.randint(-1, 2, (1,), generator=gen))
        grid = torch.roll(grid, shifts=(dy, dx), dims=(0, 1))
        # blank any row/column that wrapped around
        if dy == 1:
            grid[0, :] = 0.0
        elif dy == -1:
            grid[-1, :] = 0.0
        if dx == 1:
            grid[:, 0] = 0.0
        elif dx == -1:
            grid[:, -1] = 0.0

    if noise_p > 0:
        flips = torch.rand(grid.shape, generator=gen) < noise_p
        grid = torch.where(flips, 1.0 - grid, grid)

    return grid.reshape(-1)


def encode_character(
    ch: str,
    t_stim: int,
    rate_on: float,
    rate_off: float,
    gen: torch.Generator,
    noise_p: float = 0.0,
    shift_prob: float = 0.0,
) -> torch.Tensor:
    """Poisson-encode one character into a ``(t_stim, 35)`` binary spike tensor."""
    bitmap = char_bitmap(ch)
    if noise_p > 0.0 or shift_prob > 0.0:
        bitmap = _augment(bitmap, gen, noise_p, shift_prob)

    rate = rate_off + bitmap * (rate_on - rate_off)
    u = torch.rand((t_stim, N_PIXELS), generator=gen)
    return (u < rate.unsqueeze(0)).float()


def encode_sequence(
    text: str,
    t_stim: int,
    rate_on: float,
    rate_off: float,
    gen: torch.Generator,
    noise_p: float = 0.0,
    shift_prob: float = 0.0,
) -> Iterable[torch.Tensor]:
    for ch in text.upper():
        yield encode_character(ch, t_stim, rate_on, rate_off, gen, noise_p, shift_prob)


def encode_from_config(
    cfg,
    ch: str,
    augment: bool = False,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """Encode one character using a :class:`Config`.

    Augmentation (pixel jitter and noise) is applied only when ``augment`` is
    set, so evaluation always sees the clean glyph.
    """
    if gen is None:
        gen = torch.Generator().manual_seed(cfg.seed + 5)
    return encode_character(
        ch,
        t_stim=cfg.t_stim,
        rate_on=cfg.rate_on,
        rate_off=cfg.rate_off,
        gen=gen,
        noise_p=cfg.noise_p if augment else 0.0,
        shift_prob=cfg.shift_prob if augment else 0.0,
    )
