"""A fly mushroom body trained by dopamine-modulated Hebbian plasticity.

This is a brain-*inspired* spiking classifier, not an emulation of the FlyWire
connectome. See the README for exactly which parts are measured fly anatomy and
which parts are modelling choices.
"""

from .config import Config
from .dopamine import Burst, DopamineSystem
from .encoding import (
    CLASSES,
    FONT_5X7,
    FONT_DIGITS,
    INPUT_GLYPHS,
    char_bitmap,
    encode_character,
    encode_from_config,
    is_class,
    is_input,
    render,
)
from .circuit import AntennalLobe, KenyonCells, MushroomBody, MushroomBodyOutput
from .trainer import EpochStats, FlyBrain, TrialResult

__version__ = "0.1.0"

__all__ = [
    "Config",
    "Burst",
    "DopamineSystem",
    "CLASSES",
    "FONT_5X7",
    "FONT_DIGITS",
    "INPUT_GLYPHS",
    "char_bitmap",
    "encode_character",
    "encode_from_config",
    "is_class",
    "is_input",
    "render",
    "AntennalLobe",
    "KenyonCells",
    "MushroomBody",
    "MushroomBodyOutput",
    "EpochStats",
    "FlyBrain",
    "TrialResult",
]
