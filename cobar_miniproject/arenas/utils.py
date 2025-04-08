import numpy as np
import cv2


def get_random_pos(
    distance_range: tuple[float, float],
    angle_range: tuple[float, float],
    rng: np.random.Generator,
):
    """Generate a random target position.

    Parameters
    ----------
    distance_range : tuple[float, float]
        Distance range from the origin.
    angle_range : tuple[float, float]
        Angle rabge in radians.
    rng : np.random.Generator
        The random number generator.
    Returns
    -------
    np.ndarray
        The target position in the form of [x, y].
    """
    p = rng.uniform(*distance_range) * np.exp(1j * rng.uniform(*angle_range))
    return np.array([p.real, p.imag], float)


def circ(
    img: np.ndarray,
    xy: tuple[float, float],
    r: float,
    value: bool,
    xmin: float,
    ymin: float,
    res: float,
    outer=False,
):
    """Draw a circle on a 2D image.

    Parameters
    ----------
    img : np.ndarray
        The image to draw on.
    xy : tuple[float, float]
        The center of the circle.
    r : float
        The radius of the circle.
    value : bool
        The value to set the pixels to.
    xmin : float
        The minimum x value of the grid.
    ymin : float
        The minimum y value of the grid.
    res : float
        The resolution of the grid.
    outer : bool, optional
        If True, draw the outer circle. Otherwise, draw a filled circle.

    Returns
    -------
    None
    """
    center = ((np.asarray(xy) - (xmin, ymin)) / res).astype(int)
    radius = int(r / res) + 1 if outer else int(r / res)
    color = bool(value)
    thickness = 1 if outer else -1
    cv2.circle(img, center, radius, color, thickness)
