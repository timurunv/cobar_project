from .pillars_looming import PillarsLoomingArena


class LoomingArena(PillarsLoomingArena):
    def __init__(self, **kwargs):
        super().__init__(spawn_pillars=False, **kwargs)
