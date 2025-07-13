import random
from core.math import Ray, AABB
from core.material import HitRecord
from core.geometry import Hittable


class BVHNode(Hittable):
    def __init__(self, objects, start, end):
        axis = random.randint(0, 2)
        if axis == 0:
            objects[start:end] = sorted(objects[start:end], key=lambda o: o.bounding_box().min.x)
        elif axis == 1:
            objects[start:end] = sorted(objects[start:end], key=lambda o: o.bounding_box().min.y)
        else:
            objects[start:end] = sorted(objects[start:end], key=lambda o: o.bounding_box().min.z)

        span = end - start
        if span == 1:
            self.left = self.right = objects[start]
        elif span == 2:
            self.left = objects[start]
            self.right = objects[start + 1]
        else:
            mid = start + span // 2
            self.left = BVHNode(objects, start, mid)
            self.right = BVHNode(objects, mid, end)

        box_left = self.left.bounding_box()
        box_right = self.right.bounding_box()
        self.box = AABB.surrounding_box(box_left, box_right)

    def hit(self, ray: Ray, t_min: float, t_max: float, rec: HitRecord) -> bool:
        if not self.box.hit(ray, t_min, t_max):
            return False

        hit_left = self.left.hit(ray, t_min, t_max, rec)
        if hit_left:
            t_max = rec.t
        hit_right = self.right.hit(ray, t_min, t_max, rec)
        return hit_left or hit_right

    def bounding_box(self) -> AABB:
        return self.box