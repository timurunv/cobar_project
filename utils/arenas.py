from flygym.examples.vision.arena import ObstacleOdorArena
import cv2
import numpy as np
from flygym.arena import FlatTerrain


def get_random_target_position(
    r_range=(7, 10),
    theta_range=(-np.pi, np.pi),
    seed=None,
):
    rng = np.random.default_rng(seed)
    p = rng.uniform(*r_range) * np.exp(1j * rng.uniform(*theta_range))
    return np.array([p.real, p.imag])

def get_random_path(
    target_position,
    d_max=0.5,
    n_points=100,
    seed=None,
):
    from scipy.interpolate import CubicSpline
    rng = np.random.default_rng(seed)
    p = np.array([1, 1j]) @ target_position
    y = [0, *rng.uniform(-d_max, d_max, 2), 0]
    x = np.linspace(0, 1, 4)
    xn = np.linspace(0, 1, n_points)
    yn = CubicSpline(x, y, bc_type=((1, 0.0), "not-a-knot"))(xn)
    traj = (xn + yn * 1j) * p
    return np.column_stack([traj.real, traj.imag])


def get_walls(path, w, h, thick):
    res = 0.05
    xmin, ymin = path.min(0) - w - res
    xmax, ymax = path.max(0) + w + res
    n_cols = int((xmax - xmin) / res)
    n_rows = int((ymax - ymin) / res)
    im = np.zeros((n_rows, n_cols), dtype=np.uint8)
    line = [((path - (xmin, ymin)) / res).astype(np.int32)]
    im = cv2.polylines(im, line, isClosed=False, color=1, thickness=int(w * 2 / res))
    contour = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    is_ccw = cv2.contourArea(contour, oriented=True) > 0
    if not is_ccw:
        contour = contour[::-1]
    step = 20
    points = (contour[::step, 0] * res + (xmin, ymin)) @ (1, 1j)
    diff = np.roll(points, -1) - points
    rx = np.abs(diff) / 2
    theta_z = np.angle(diff)
    centers = (np.roll(points, -1) + points) / 2 + diff / np.abs(diff) * -1j * thick
    
    for rx_, theta_z_, center_ in zip(rx, theta_z, centers):
        yield dict(
            element_name="geom",
            type="box",
            size=(rx_, thick, h),
            pos=(center_.real, center_.imag, h),
            rgba=(1, 0, 0, 1),
            euler=(0, 0, theta_z_),
        )


def get_pillars_for_corridor(path, w, thick, r, sep):
    import cv2

    res = 0.05
    xmin, ymin = path.min(0) - w - thick + r - res
    xmax, ymax = path.max(0) + w + thick - r + res
    n_cols = int((xmax - xmin) / res)
    n_rows = int((ymax - ymin) / res)
    im = np.zeros((n_rows, n_cols), dtype=np.uint8)
    line = [((path - (xmin, ymin)) / res).astype(np.int32)]
    im_inner = cv2.polylines(
        im.copy(), line, isClosed=False, color=1, thickness=int((w + r) * 2 / res),
    )

    im_outer = cv2.polylines(
        im.copy(), line, isClosed=False, color=1, thickness=int((w + thick - r) * 2 / res),
    )

    im = im_outer & ~im_inner
    
    pos = []
    while True:
        argwhere = np.argwhere(im)
        
        try:
            p = argwhere[np.random.choice(len(argwhere))]
        except ValueError:
            break
        
        cv2.circle(im, (p[1], p[0]), int((r + sep) / res), color=0, thickness=-1)
        pos.append((p[1] * res + xmin, p[0] * res + ymin))
    
    return np.array(pos)


class Corridor(ObstacleOdorArena):
    def __init__(
        self,
        h=2,
        w=3,
        r_range=(19, 20),
        theta_range=(-np.pi / 6, np.pi / 6),
        d_max=0.25,
        thick=0.1,
        marker_size=0.3,
        seed=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        h : float
            Height of the walls.
        w : float
            Half width of the corridor.
        thick : float
            Thickness of the walls.
        d_max : float
            Maximum deviation of the path from the straight line.
        """
        target_position = get_random_target_position(
            r_range=r_range,
            theta_range=theta_range,
            seed=seed,
        )
        path = get_random_path(
            target_position=target_position,
            d_max=d_max,
            seed=seed,
        )
        super().__init__(
            terrain=FlatTerrain(ground_alpha=0),
            odor_source=np.array([[*target_position, 1]]),
            marker_colors=np.array([[255, 127, 14, 255]]) / 255,
            peak_odor_intensity=np.array([[1, 0]]),
            obstacle_positions=np.array([]),
            marker_size=marker_size,
            **kwargs,
        )
        for wall_params in get_walls(path, w, h, thick):
            self.root_element.worldbody.add(**wall_params)

class CorridorWithPillars(ObstacleOdorArena):
    def __init__(
        self,
        h=3,
        w=3,
        r_range=(19, 20),
        theta_range=(-np.pi / 6, np.pi / 6),
        d_max=0.25,
        thick=0.5,
        r=0.2,
        sep=1,
        marker_size=0.3,
        seed=None,
        **kwargs,
    ):
        target_position = get_random_target_position(
            r_range=r_range,
            theta_range=theta_range,
            seed=seed,
        )
        path = get_random_path(
            target_position=target_position,
            d_max=d_max,
            seed=seed,
        )

        pos = get_pillars_for_corridor(path, w, thick, r, sep)
        
        super().__init__(
            odor_source=np.array([[*target_position, 1]]),
            marker_colors=np.array([[255, 127, 14, 255]]) / 255,
            peak_odor_intensity=np.array([[1, 0]]),
            obstacle_positions=pos,
            obstacle_radius=r,
            obstacle_height=h,
            terrain=FlatTerrain(ground_alpha=0),
            marker_size=marker_size,
            **kwargs,
        )


def circ(im, pos, r, value, xmin, ymin, res):
    import cv2
    c = ((np.asarray(pos) - (xmin, ymin)) / res).astype(int)
    r = int(r / res)
    cv2.circle(im, center=c, radius=r, color=value, thickness=-1)


class ScatteredPillarsArena(ObstacleOdorArena):
    def __init__(
        self,
        r_range=(19, 20),
        theta_range=(-np.pi / 6, np.pi / 6),
        pillar_height=3,
        pillar_radius=0.2,
        pillar_separation=4,
        target_space=4,
        fly_space=4,
        res=0.05,
        target_size=0.3,
        target_color=(1, 0.5, 14 / 255, 1),
        seed=None,
        **kwargs,
    ):
        self.target_space = target_space
        self.fly_space = fly_space
        self.pillar_radius = pillar_radius
        self.pillar_height = pillar_height
        self.pillar_separation = pillar_separation
        self.target_size = target_size
        self.target_color = target_color

        target_position = get_random_target_position(
            r_range=r_range,
            theta_range=theta_range,
            seed=seed,
        )

        pillar_positions = self.get_pillar_positions(
            target_position,
            pillar_radius=pillar_radius,
            target_space=target_space,
            pillar_separation=pillar_separation,
            fly_space=fly_space,
            res=res,
            seed=seed,
        )
        
        super().__init__(
            odor_source=np.array([[*target_position, 1]]),
            marker_colors=np.array([target_color]),
            peak_odor_intensity=np.array([[1, 0]]),
            obstacle_positions=pillar_positions,
            obstacle_radius=pillar_radius,
            obstacle_height=pillar_height,
            terrain=FlatTerrain(ground_alpha=0),
            marker_size=target_size,
            **kwargs,
        )

    @staticmethod
    def get_pillar_positions(
        target_position,
        pillar_radius=0.2,
        target_space=4,
        pillar_separation=4,
        fly_space=4,
        res=0.05,
        seed=None,
    ):
        pillar_space = pillar_radius + pillar_separation
        vmax = np.linalg.norm(target_position)
        xmin = ymin = -vmax
        xmax = ymax = vmax
        n_cols = int((xmax - xmin) / res)
        n_rows = int((ymax - ymin) / res)
        im = np.zeros((n_rows, n_cols), dtype=np.uint8)
        circ(im, (0, 0), vmax, 255, xmin, ymin, res)
        circ(im, (0, 0), fly_space, 0, xmin, ymin, res)
        circ(im, target_position, target_space, 0, xmin, ymin, res)
        pillar0_position = target_position / 2
        circ(im, pillar0_position, pillar_space, 0, xmin, ymin, res)

        rng = np.random.default_rng(seed)
        pillar_positions = [pillar0_position]

        while True:
            argwhere = np.argwhere(im)
            try:
                p = argwhere[rng.choice(len(argwhere)), ::-1] * res + (xmin, ymin)
            except ValueError:
                break
            pillar_positions.append(p)
            circ(im, p, pillar_space, 0, xmin, ymin, res)
        
        return np.array(pillar_positions)
