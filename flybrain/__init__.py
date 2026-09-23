"""A fly mushroom body trained by dopamine-modulated Hebbian plasticity.

This is a brain-*inspired* spiking classifier, not an emulation of a fly brain.
As of v0.2.0 the fixed ``PN -> KC`` expansion can be taken from the measured
hemibrain connectome instead of a random number generator (``Config.wiring``),
which makes the wiring real while leaving the task invented. See the README for
exactly which parts are measured fly anatomy and which are modelling choices.
"""

from .config import Config
from .connectome import Connectome, pn_kc_weights
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

__version__ = "0.2.0"

__all__ = [
    "Config",
    "Connectome",
    "pn_kc_weights",
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
