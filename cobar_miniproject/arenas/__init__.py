from .odor import OdorArena
from .pillars import PillarsArena
from .pillars_looming import PillarsLoomingArena
from .looming import LoomingArena

level_arenas = {
    1: OdorArena,
    2: PillarsArena,
    3: LoomingArena,
    4: PillarsLoomingArena,
}
