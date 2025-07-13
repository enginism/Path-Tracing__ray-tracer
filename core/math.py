import math
import numpy as np

class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other):
        return Vec3(self.x + other.x,
                    self.y + other.y,
                    self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x,
                    self.y - other.y,
                    self.z - other.z)

    def __mul__(self, t):
        # 스칼라 곱 또는 원소별 곱(Hadamard)
        if isinstance(t, Vec3):
            return Vec3(self.x * t.x,
                        self.y * t.y,
                        self.z * t.z)
        return Vec3(self.x * t, self.y * t, self.z * t)

    __rmul__ = __mul__

    def __truediv__(self, t):
        return Vec3(self.x / t, self.y / t, self.z / t)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        l = self.length()
        if l == 0:
            return Vec3(0, 0, 0)
        return self / l

    def reflect(self, normal):
        # 반사 벡터: r = v - 2 * dot(v, n) * n
        return self - normal * (2 * self.dot(normal))

    def refract(self, normal, ni_over_nt):
        uv = self.normalize()
        dt = uv.dot(normal)
        discr = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt)
        if discr > 0:
            refracted = (uv - normal * dt) * ni_over_nt - normal * math.sqrt(discr)
            return True, refracted
        else:
            return False, None

    def to_np(self):
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def __repr__(self):
        return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


class Ray:
    def __init__(self, origin: Vec3, direction: Vec3):
        self.origin = origin
        self.direction = direction.normalize()

    def point_at_parameter(self, t):
        return self.origin + self.direction * t


class AABB:
    def __init__(self, min_pt: Vec3, max_pt: Vec3):
        self.min = min_pt
        self.max = max_pt

    @staticmethod
    def surrounding_box(box0, box1):
        small = Vec3(
            min(box0.min.x, box1.min.x),
            min(box0.min.y, box1.min.y),
            min(box0.min.z, box1.min.z)
        )
        big = Vec3(
            max(box0.max.x, box1.max.x),
            max(box0.max.y, box1.max.y),
            max(box0.max.z, box1.max.z)
        )
        return AABB(small, big)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> bool:
        for a in range(3):
            invD = 1.0 / (ray.direction.x if a == 0 else (ray.direction.y if a == 1 else ray.direction.z))
            t0 = ((self.min.x if a == 0 else (self.min.y if a == 1 else self.min.z)) -
                  (ray.origin.x if a == 0 else (ray.origin.y if a == 1 else ray.origin.z))) * invD
            t1 = ((self.max.x if a == 0 else (self.max.y if a == 1 else self.max.z)) -
                  (ray.origin.x if a == 0 else (ray.origin.y if a == 1 else ray.origin.z))) * invD
            if invD < 0.0:
                t0, t1 = t1, t0
            t_min = t0 if t0 > t_min else t_min
            t_max = t1 if t1 < t_max else t_max
            if t_max < t_min:
                return False
        return True