import math
import random
import time
from typing import List
from PIL import Image

from core.math import Vec3, Ray
from core.material import HitRecord
from core.scene import Scene, RenderSettings
from core.camera import Camera
from renderers.base_renderer import BaseRenderer, RendererFactory


class CPURenderer(BaseRenderer):
    """CPU 기반 레이트레이싱 렌더러"""

    def __init__(self):
        super().__init__("cpu_raytracer")

    def get_capabilities(self) -> List[str]:
        return [
            "ray_tracing",
            "shadows",
            "reflection",
            "refraction",
            "area_lights",
            "anti_aliasing",
            "bvh_acceleration"
        ]

    def render(self, scene: Scene, camera: Camera, settings: RenderSettings) -> Image.Image:
        """메인 렌더링 함수"""
        start_time = time.time()

        print(f"CPU 렌더링 시작: {settings.width}x{settings.height}, {settings.samples_per_pixel} samples")

        output_image = Image.new("RGB", (settings.width, settings.height))
        pixels = output_image.load()

        grid_n = int(math.sqrt(settings.samples_per_pixel))

        for j in range(settings.height):
            for i in range(settings.width):
                col = Vec3(0, 0, 0)

                # 픽셀을 grid_n x grid_n 격자로 나누고, 각 셀마다 Jittered 샘플링
                for a in range(grid_n):
                    for b in range(grid_n):
                        du = (a + random.random()) / grid_n
                        dv = (b + random.random()) / grid_n

                        u = (i + du) / settings.width
                        v = (j + dv) / settings.height

                        ray = camera.get_ray(u, v)
                        col += self._trace(ray, scene, 0, settings.max_depth)

                col /= settings.samples_per_pixel
                r = int(max(0, min(255, col.x * 255)))
                g = int(max(0, min(255, col.y * 255)))
                b = int(max(0, min(255, col.z * 255)))
                pixels[i, settings.height - 1 - j] = (r, g, b)

            if j % 50 == 0:
                print(f"CPU is working for you...: {settings.height - j}")

        end_time = time.time()
        elapsed = end_time - start_time
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"CPU 렌더링 완료: {minutes}분 {seconds:.2f}초")

        return output_image

    def _trace(self, ray: Ray, scene: Scene, depth: int, max_depth: int) -> Vec3:
        """레이트레이싱 함수"""
        rec = HitRecord()
        if scene.hit(ray, 1e-3, float('inf'), rec):
            mat = rec.material

            # 베이스 컬러: 텍스처 있으면 샘플링, 없으면 material.color
            if mat.texture is not None:
                base_color = mat.texture.sample(rec.u, rec.v)
            else:
                base_color = mat.color

            # 1) Ambient 기여
            local_color = mat.diffuse * base_color * scene.ambient

            # 2) Area Light 샘플마다 Diffuse + Specular + Shadow (Monte Carlo 평균)
            n_samples = len(scene.lights)
            for light_pos in scene.lights:
                to_light = (light_pos - rec.point).normalize()
                shadow_ray = Ray(rec.point + rec.normal * 1e-3, to_light)
                shadow_rec = HitRecord()
                in_shadow = False
                dist_to_light = (light_pos - rec.point).length()
                if scene.hit(shadow_ray, 1e-3, dist_to_light, shadow_rec):
                    in_shadow = True

                if not in_shadow:
                    # Diffuse (Lambert)
                    diff = max(rec.normal.dot(to_light), 0.0)
                    local_color += (mat.diffuse * base_color * scene.light_color * diff) / n_samples

                    # Specular (Phong)
                    view_dir = (ray.origin - rec.point).normalize()
                    reflect_dir = to_light.reflect(rec.normal)
                    spec = max(view_dir.dot(reflect_dir), 0.0)
                    local_color += (mat.specular * (spec ** 32) * scene.light_color) / n_samples

            # 3) Reflection
            reflected_color = Vec3(0, 0, 0)
            if mat.reflective > 0 and depth < max_depth:
                reflected_dir = ray.direction.reflect(rec.normal)
                reflected_ray = Ray(rec.point + rec.normal * 1e-3, reflected_dir)
                reflected_color = self._trace(reflected_ray, scene, depth + 1, max_depth)

            # 4) Refraction
            refracted_color = Vec3(0, 0, 0)
            if mat.refractive > 0 and depth < max_depth:
                outward_normal = None
                ni_over_nt = 0.0
                cosine = 0.0
                if ray.direction.dot(rec.normal) > 0:
                    outward_normal = -rec.normal
                    ni_over_nt = mat.ior
                    cosine = mat.ior * ray.direction.dot(rec.normal) / ray.direction.length()
                else:
                    outward_normal = rec.normal
                    ni_over_nt = 1.0 / mat.ior
                    cosine = -ray.direction.dot(rec.normal) / ray.direction.length()

                did_refract, refracted_dir = ray.direction.refract(outward_normal, ni_over_nt)
                if did_refract:
                    refracted_ray = Ray(rec.point - rec.normal * 1e-3, refracted_dir)
                    refracted_color = self._trace(refracted_ray, scene, depth + 1, max_depth)
                else:
                    # Total internal reflection
                    reflected_dir = ray.direction.reflect(rec.normal)
                    reflected_ray = Ray(rec.point + rec.normal * 1e-3, reflected_dir)
                    refracted_color = self._trace(reflected_ray, scene, depth + 1, max_depth)

            color = Vec3(0, 0, 0)
            color += local_color * (1.0 - mat.reflective - mat.refractive)
            color += reflected_color * mat.reflective
            color += refracted_color * mat.refractive
            return color

        # 배경: 검은색
        return Vec3(0, 0, 0)


# 렌더러 등록
RendererFactory.register("cpu_raytracer", CPURenderer)