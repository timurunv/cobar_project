from .arenas import (
    OdorTargetOnlyArena,
    ScatteredPillarsArena,
    LoomingBallArena,
    HierarchicalArena,
    FoodToNestArena,
)

level_arenas = {
    1: OdorTargetOnlyArena,
    2: ScatteredPillarsArena,
    3: LoomingBallArena,
    4: HierarchicalArena,
    5: FoodToNestArena,
}
