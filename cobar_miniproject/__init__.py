from .base_controller import BaseController
from .arenas import ScatteredPillarsArena

__all__ = ["BaseController"]
levels = {
    0: ScatteredPillarsArena,
    1: ScatteredPillarsArena,
    2: ScatteredPillarsArena,
    3: ScatteredPillarsArena,
    4: ScatteredPillarsArena,
}
