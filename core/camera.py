import math
from core.math import Vec3, Ray


class Camera:
    def __init__(self,
                 lookfrom: Vec3,
                 lookat: Vec3,
                 vup: Vec3,
                 vfov: float,        # 수직 FOV(deg)
                 aspect: float):     # 가로/세로 비율
        self.origin = lookfrom

        theta = math.radians(vfov)
        half_height = math.tan(theta / 2)
        half_width = aspect * half_height

        w = (lookfrom - lookat).normalize()
        u = vup.cross(w).normalize()
        v = w.cross(u)

        self.lower_left_corner = self.origin - u * half_width - v * half_height - w
        self.horizontal = u * (2 * half_width)
        self.vertical = v * (2 * half_height)

    def get_ray(self, s: float, t: float) -> Ray:
        direction = (self.lower_left_corner +
                     self.horizontal * s +
                     self.vertical * t -
                     self.origin)
        return Ray(self.origin, direction)