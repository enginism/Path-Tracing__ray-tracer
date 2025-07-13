import random
from typing import List
from dataclasses import dataclass
from core.math import Vec3, Ray
from core.material import HitRecord
from core.geometry import Hittable
from core.acceleration import BVHNode


@dataclass
class CameraParams:
    lookfrom: Vec3
    lookat: Vec3
    vup: Vec3
    vfov: float
    aspect: float


@dataclass
class RenderSettings:
    width: int = 800
    height: int = 600
    samples_per_pixel: int = 9
    max_depth: int = 4


class Scene:
    def __init__(self):
        self.objects: List[Hittable] = []
        self.bvh_root = None
        self.lights: List[Vec3] = []
        self.light_color = Vec3(1.0, 1.0, 1.0)
        self.ambient = Vec3(0.5, 0.5, 0.5)

    def add_object(self, obj: Hittable):
        self.objects.append(obj)

    def build_bvh(self):
        if len(self.objects) > 0:
            self.bvh_root = BVHNode(self.objects, 0, len(self.objects))

    def add_light_sample(self, pos: Vec3):
        self.lights.append(pos)

    def hit(self, ray: Ray, t_min: float, t_max: float, rec: HitRecord) -> bool:
        if self.bvh_root:
            return self.bvh_root.hit(ray, t_min, t_max, rec)
        else:
            temp_rec = HitRecord()
            hit_anything = False
            closest_so_far = t_max

            for obj in self.objects:
                if obj.hit(ray, t_min, closest_so_far, temp_rec):
                    hit_anything = True
                    closest_so_far = temp_rec.t
                    rec.t = temp_rec.t
                    rec.point = temp_rec.point
                    rec.normal = temp_rec.normal
                    rec.material = temp_rec.material
                    rec.u = temp_rec.u
                    rec.v = temp_rec.v

            return hit_anything


def create_area_light(scene: Scene,
                      center: Vec3,
                      u_vec: Vec3, v_vec: Vec3,
                      u_size: float, v_size: float,
                      n_u: int, n_v: int):
    """Area Light 생성: 천장에 n_u×n_v 그리드로 샘플 배치"""
    half_u = u_vec.normalize() * (u_size / 2.0)
    half_v = v_vec.normalize() * (v_size / 2.0)
    for i in range(n_u):
        for j in range(n_v):
            ru = (i + 0.5) / n_u - 0.5
            rv = (j + 0.5) / n_v - 0.5
            sample_pos = center + half_u * (2 * ru) + half_v * (2 * rv)
            scene.add_light_sample(sample_pos)