import numpy as np
from PIL import Image
from core.math import Vec3


class Texture:
    def __init__(self, path: str):
        self.path = path  # 텍스처 파일 경로 저장
        img = Image.open(path).convert("RGB")
        self.width, self.height = img.size
        self.pixels = np.array(img)  # (height, width, 3)

    def sample(self, u: float, v: float) -> Vec3:
        """
        u, v ∈ [0,1], (0,0)이 좌상단, (1,1)이 우하단.
        PIL 이미지 v축은 위→아래 방향이라 (1-v)로 뒤집어서 인덱싱.
        """
        iu = int(max(0, min(self.width - 1, u * (self.width - 1))))
        iv = int(max(0, min(self.height - 1, (1.0 - v) * (self.height - 1))))
        r, g, b = self.pixels[iv, iu]
        return Vec3(r / 255.0, g / 255.0, b / 255.0)


class Material:
    def __init__(self,
                 color: Vec3 = Vec3(1, 1, 1),
                 diffuse=1.0,
                 specular=0.0,
                 reflective=0.0,
                 refractive=0.0,
                 ior=1.0,
                 texture: Texture = None):
        """
        color: Vec3, 텍스처 없을 때 사용될 고정 색
        diffuse: 확산 Lambertian 계수
        specular: Phong 스페큘러 계수
        reflective: 반사 강도 (0~1)
        refractive: 굴절 강도 (0~1)
        ior: 굴절률(Index of Refraction)
        texture: 텍스처 이미지가 있으면 (u, v)로부터 샘플링
        """
        self.color = color
        self.diffuse = diffuse
        self.specular = specular
        self.reflective = reflective
        self.refractive = refractive
        self.ior = ior
        self.texture = texture


class HitRecord:
    def __init__(self):
        self.t = float('inf')
        self.point = None
        self.normal = None
        self.material = None
        self.u = 0.0
        self.v = 0.0