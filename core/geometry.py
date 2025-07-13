import math
import numpy as np
from abc import ABC, abstractmethod
from core.math import Vec3, Ray, AABB
from core.material import Material, HitRecord


class Hittable(ABC):
    @abstractmethod
    def hit(self, ray: Ray, t_min: float, t_max: float, rec: HitRecord) -> bool:
        pass

    @abstractmethod
    def bounding_box(self) -> AABB:
        pass


class Plane(Hittable):
    def __init__(self,
                 anchor: Vec3,        # 평면 위의 한 점 (corner)
                 normal: Vec3,        # 법선
                 u_dir: Vec3,         # 텍스처 u 축 벡터
                 v_dir: Vec3,         # 텍스처 v 축 벡터
                 u_len: float,        # u 축 실제 길이 (world 단위)
                 v_len: float,        # v 축 실제 길이 (world 단위)
                 material: Material):
        self.anchor = anchor
        self.normal = normal.normalize()
        self.u_dir = u_dir
        self.v_dir = v_dir
        self.u_len = u_len
        self.v_len = v_len
        self.material = material

        self.u_unit = u_dir.normalize()
        self.v_unit = self.normal.cross(self.u_unit).normalize()
        self.u_extent = u_len
        self.v_extent = v_len

        # AABB 계산을 위해 4개의 모서리 점 저장
        corner0 = anchor
        corner1 = anchor + self.u_unit * u_len
        corner2 = anchor + self.v_unit * v_len
        corner3 = anchor + self.u_unit * u_len + self.v_unit * v_len
        xs = [corner0.x, corner1.x, corner2.x, corner3.x]
        ys = [corner0.y, corner1.y, corner2.y, corner3.y]
        zs = [corner0.z, corner1.z, corner2.z, corner3.z]
        self.box = AABB(Vec3(min(xs), min(ys), min(zs)), Vec3(max(xs), max(ys), max(zs)))

    def hit(self, ray: Ray, t_min: float, t_max: float, rec: HitRecord) -> bool:
        denom = self.normal.dot(ray.direction)
        if abs(denom) < 1e-6:
            return False  # 광선과 평면이 평행

        t = (self.anchor - ray.origin).dot(self.normal) / denom
        if t < t_min or t > t_max:
            return False

        P = ray.point_at_parameter(t)
        v2 = P - self.anchor
        u_hit = v2.dot(self.u_unit)
        v_hit = v2.dot(self.v_unit)
        if u_hit < 0 or u_hit > self.u_extent or v_hit < 0 or v_hit > self.v_extent:
            return False

        rec.t = t
        rec.point = P
        rec.normal = self.normal
        rec.material = self.material
        rec.u = u_hit / self.u_extent
        rec.v = v_hit / self.v_extent
        return True

    def bounding_box(self) -> AABB:
        return self.box


class Sphere(Hittable):
    def __init__(self, center: Vec3, radius: float, material: Material):
        self.center = center
        self.radius = radius
        self.material = material
        self.box = AABB(center - Vec3(radius, radius, radius), center + Vec3(radius, radius, radius))

    def hit(self, ray: Ray, t_min: float, t_max: float, rec: HitRecord) -> bool:
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - a * c
        if discriminant > 0:
            sqrt_d = math.sqrt(discriminant)
            temp = (-b - sqrt_d) / a
            if t_min < temp < t_max:
                rec.t = temp
                rec.point = ray.point_at_parameter(rec.t)
                rec.normal = (rec.point - self.center) / self.radius
                rec.material = self.material
                rec.u = 0.0
                rec.v = 0.0
                return True
            temp = (-b + sqrt_d) / a
            if t_min < temp < t_max:
                rec.t = temp
                rec.point = ray.point_at_parameter(rec.t)
                rec.normal = (rec.point - self.center) / self.radius
                rec.material = self.material
                rec.u = 0.0
                rec.v = 0.0
                return True
        return False

    def bounding_box(self) -> AABB:
        return self.box


class Triangle(Hittable):
    def __init__(self,
                 v0: Vec3, v1: Vec3, v2: Vec3,
                 uv0: np.ndarray = None, uv1: np.ndarray = None, uv2: np.ndarray = None,
                 material: Material = None):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.material = material
        self.uv0 = uv0
        self.uv1 = uv1
        self.uv2 = uv2
        self.normal = (v1 - v0).cross(v2 - v0).normalize()
        # AABB 계산
        min_x = min(v0.x, v1.x, v2.x)
        min_y = min(v0.y, v1.y, v2.y)
        min_z = min(v0.z, v1.z, v2.z)
        max_x = max(v0.x, v1.x, v2.x)
        max_y = max(v0.y, v1.y, v2.y)
        max_z = max(v0.z, v1.z, v2.z)
        self.box = AABB(Vec3(min_x, min_y, min_z), Vec3(max_x, max_y, max_z))

    def hit(self, ray: Ray, t_min: float, t_max: float, rec: HitRecord) -> bool:
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        h = ray.direction.cross(edge2)
        a = edge1.dot(h)
        if abs(a) < 1e-6:
            return False  # 평행

        f = 1.0 / a
        s = ray.origin - self.v0
        u = f * s.dot(h)
        if u < 0.0 or u > 1.0:
            return False

        q = s.cross(edge1)
        v = f * ray.direction.dot(q)
        if v < 0.0 or u + v > 1.0:
            return False

        t = f * edge2.dot(q)
        if t_min < t < t_max:
            rec.t = t
            rec.point = ray.point_at_parameter(t)
            rec.normal = self.normal if self.normal.dot(ray.direction) < 0 else -self.normal
            rec.material = self.material
            if self.uv0 is not None:
                w_bary = 1 - u - v
                rec.u = u * self.uv1[0] + v * self.uv2[0] + w_bary * self.uv0[0]
                rec.v = u * self.uv1[1] + v * self.uv2[1] + w_bary * self.uv0[1]
            else:
                rec.u, rec.v = 0.0, 0.0
            return True
        return False

    def bounding_box(self) -> AABB:
        return self.box