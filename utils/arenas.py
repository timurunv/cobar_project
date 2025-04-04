import cv2
import numpy as np
from flygym.arena import FlatTerrain

def get_random_path(
    r_range=(7, 10),
    theta_range=(-np.pi, np.pi),
    d_max=0.5,
    n_points=100,
    seed=None,
):
    from scipy.interpolate import CubicSpline
    rng = np.random.default_rng(seed)
    p = rng.uniform(*r_range) * np.exp(1j * rng.uniform(*theta_range))
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


class Corridor(FlatTerrain):
    def __init__(
        self,
        h=2,
        w=3,
        r_range=(19, 20),
        theta_range=(-np.pi / 6, np.pi / 6),
        d_max=0.25,
        thick=0.1,
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
        super().__init__(**kwargs, ground_alpha=0)
        path = get_random_path(
            r_range=r_range,
            theta_range=theta_range,
            d_max=d_max,
            seed=None,
        )
        
        for wall_params in get_walls(path, w, h, thick):
            self.root_element.worldbody.add(**wall_params)
