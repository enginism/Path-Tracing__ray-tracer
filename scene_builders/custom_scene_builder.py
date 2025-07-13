import math
import numpy as np
from core.math import Vec3
from core.material import Material, Texture
from core.geometry import Plane, Sphere, Triangle
from core.scene import Scene, create_area_light
from core.camera import Camera


class CustomSceneBuilder:
    """당신의 실제 코넬박스를 재현하는 빌더 (굴절 재질 포함)"""

    def __init__(self):
        # 실제 치수 (cm 단위)
        self.box_size = 30.0  # 30x30x30cm 내부 크기
        self.foam_thickness = 0.5  # 폼보드 두께

        # 루빅스 큐브
        self.cube_size = 5.6  # 5.6x5.6x5.6cm

        # 캔버스
        self.canvas_width = 27.5  # 가로
        self.canvas_height = 22.0  # 세로
        self.canvas_depth = 1.5  # 두께
        self.canvas_angle = 112.0  # 기울기 각도

        # 조명
        self.light_size = 3.0  # 4x4cm 조명구멍

    def build_scene(self) -> Scene:
        """완전한 커스텀 코넬박스 씬을 생성"""
        scene = Scene()

        # 재질 생성
        materials = self._create_materials()

        # 벽들 생성 (30x30x30cm 박스)
        self._create_walls(scene, materials)

        # 루빅스 큐브 2개 배치
        self._create_rubiks_cubes(scene, materials)

        # 구체들 추가 (굴절 구체 포함)
        self._create_spheres(scene, materials)

        # 캔버스 배치
        self._create_canvas(scene, materials)

        # 조명 생성 (4x4cm 구멍)
        self._create_lighting(scene)

        # BVH 빌드
        scene.build_bvh()

        # 조명 강도 증가
        scene.light_color = Vec3(0.7, 0.7, 0.7)  # 기본값 1.0
        scene.ambient = Vec3(0.5, 0.5, 0.5)  # 기본값 0.5에서 0.3으로 약간 감소 (상대적 균형)

        return scene

    def create_camera(self, aspect_ratio: float = 4.0 / 3.0) -> Camera:
        """실제 아이폰 12 Pro 카메라 설정 재현"""
        # 박스 정중앙에서 정면으로 50cm 떨어진 위치
        lookfrom = Vec3(0, 0, 50.0)
        lookat = Vec3(0, 0, 0)  # 박스 중심을 바라봄
        vup = Vec3(0, 1, 0)

        # 아이폰 12 Pro 가로모드: 수직 FOV 49.5도
        vfov = 49.5

        return Camera(lookfrom, lookat, vup, vfov, aspect_ratio)

    def _create_materials(self) -> dict:
        """모든 재질들을 생성 (굴절 재질 포함)"""
        # 루빅스 큐브 텍스처들
        cube_textures = {
            'blue': Texture("textures/blue.jpg"),
            'green': Texture("textures/green.jpg"),
            'orange': Texture("textures/orange.jpg"),
            'red': Texture("textures/red.jpg"),
            'white': Texture("textures/white.jpg"),
            'yellow': Texture("textures/yellow.jpg"),
        }

        # 캔버스 텍스처
        canvas_texture = Texture("textures/meinsf.jpg")

        # 재질 생성
        materials = {
            # 벽 재질들
            'floor': Material(
                color=Vec3(0.9, 0.9, 0.9), diffuse=0.8, specular=0.1
            ),
            'back': Material(
                color=Vec3(0.9, 0.9, 0.9), diffuse=0.8, specular=0.1
            ),
            'left': Material(  # 핫핑크 (255, 105, 180)
                color=Vec3(255 / 255, 105 / 255, 180 / 255), diffuse=0.8, specular=0.1
            ),
            'right': Material(  # 파란색 (52, 157, 204)
                color=Vec3(52 / 255, 157 / 255, 204 / 255), diffuse=0.8, specular=0.1
            ),
            'ceiling': Material(
                color=Vec3(0.9, 0.9, 0.9), diffuse=0.8, specular=0.1
            ),

            # 루빅스 큐브 - 플라스틱 느낌 강화
            'cube_blue': Material(
                color=Vec3(0.0, 0.2, 0.8),
                diffuse=0.7,  # 0.8 → 0.7
                specular=0.4,  # 0.3 → 0.4 (플라스틱 하이라이트)
                reflective=0.0,
                texture=cube_textures['blue'],

            ),
            # 다른 큐브 면들도 동일하게...
            'cube_green': Material(
                color=Vec3(0.0, 0.6, 0.0), diffuse=0.7, specular=0.4,
                reflective=0.0, texture=cube_textures['green']
            ),
            'cube_orange': Material(
                color=Vec3(1.0, 0.4, 0.0), diffuse=0.7, specular=0.4,
                reflective=0.0, texture=cube_textures['orange']
            ),
            'cube_red': Material(
                color=Vec3(0.8, 0.0, 0.0), diffuse=0.7, specular=0.4,
                reflective=0.0, texture=cube_textures['red']
            ),
            'cube_white': Material(
                color=Vec3(0.9, 0.9, 0.9), diffuse=0.7, specular=0.4,
                reflective=0.0, texture=cube_textures['white']
            ),
            'cube_yellow': Material(
                color=Vec3(1.0, 0.9, 0.0), diffuse=0.7, specular=0.4,
                reflective=0.0, texture=cube_textures['yellow']
            ),

            # 캔버스 재질 - 따뜻한 캔버스 색상
            'canvas': Material(
                color=Vec3(0.9, 0.8, 0.6), diffuse=0.9, specular=0.1,
                texture=canvas_texture
            ),

            # 구체 재질들 - 스페큘러 강화
            'sphere_red': Material(
                color=Vec3(1, 0, 0),
                diffuse=0.7,  # 0.8 → 0.7 (조금 감소)
                specular=0.5,  # 0.3 → 0.5 (하이라이트 강화)
                reflective=0.1
            ),
            # 'sphere_metal': Material(
            #     color=Vec3(0.7, 0.7, 0.7),
            #     diffuse=0.1,
            #     specular=0.8,  # 0.9 → 0.8 (금속 하이라이트)
            #     reflective=0.8
            # ),

            'sphere_metal': Material(
                color=Vec3(0.9, 0.9, 0.9),  # 더 밝은 은색
                diffuse=0.05,  # 확산반사 최소 (매트함 제거)
                specular=0.95,  # 스페큘러 최대 (반짝임 극대화)
                reflective=0.95  # 반사율 최대 (거울 효과)
            ),

            # ========== 굴절 재질들 추가 ==========
            # 'glass': Material(
            #     color=Vec3(0.95, 0.95, 0.95),
            #     diffuse=0.0,
            #     specular=0.05,
            #     reflective=0.1,
            #     refractive=0.9,  # 높은 굴절률
            #     ior=1.5  # 유리의 굴절률
            # ),
            # 'crystal': Material(
            #     color=Vec3(0.9, 0.95, 1.0),
            #     diffuse=0.05,
            #     specular=0.1,
            #     reflective=0.15,
            #     refractive=0.8,
            #     ior=2.4  # 다이아몬드 굴절률
            # ),
            # 'water_sphere': Material(
            #     color=Vec3(0.8, 0.9, 1.0),
            #     diffuse=0.1,
            #     specular=0.2,
            #     reflective=0.1,
            #     refractive=0.7,
            #     ior=1.33  # 물의 굴절률
            # )
            # 굴절 재질들 - 약간의 스페큘러 추가
            'glass': Material(
                color=Vec3(0.95, 0.95, 0.95),
                diffuse=0.1,
                specular=0.9,     # 0.05 → 0.2 (유리 하이라이트)
                reflective=0.1,
                refractive=0.85,
                ior=1.5
            ),
            'crystal': Material(
                color=Vec3(0.9, 0.95, 1.0),
                diffuse=0.1,
                specular=0.3,     # 0.1 → 0.3 (크리스탈 하이라이트)
                reflective=0.1,
                refractive=0.8,
                ior=2.4
            ),
            'water_sphere': Material(
                color=Vec3(0.8, 0.9, 1.0),
                diffuse=0.15,
                specular=0.4,     # 0.2 → 0.4 (물방울 하이라이트)
                reflective=0.05,
                refractive=0.8,
                ior=1.33
            ),
        }

        return materials

    def _create_walls(self, scene: Scene, materials: dict):
        """30x30x30cm 코넬박스의 벽들을 생성"""
        half_size = self.box_size / 2.0  # 15cm

        # Floor: y = -15
        floor_anchor = Vec3(-half_size, -half_size, half_size)
        floor_normal = Vec3(0, 1, 0)
        floor_u_dir = Vec3(self.box_size, 0, 0)
        floor_v_dir = Vec3(0, 0, -self.box_size)
        plane_floor = Plane(
            anchor=floor_anchor, normal=floor_normal,
            u_dir=floor_u_dir, v_dir=floor_v_dir,
            u_len=self.box_size, v_len=self.box_size,
            material=materials['floor']
        )
        scene.add_object(plane_floor)

        # Back Wall: z = -15
        back_anchor = Vec3(-half_size, -half_size, -half_size)
        back_normal = Vec3(0, 0, 1)
        back_u_dir = Vec3(self.box_size, 0, 0)
        back_v_dir = Vec3(0, self.box_size, 0)
        plane_back = Plane(
            anchor=back_anchor, normal=back_normal,
            u_dir=back_u_dir, v_dir=back_v_dir,
            u_len=self.box_size, v_len=self.box_size,
            material=materials['back']
        )
        scene.add_object(plane_back)

        # Left Wall: x = -15 (핫핑크)
        left_anchor = Vec3(-half_size, -half_size, half_size)
        left_normal = Vec3(1, 0, 0)
        left_u_dir = Vec3(0, 0, -self.box_size)
        left_v_dir = Vec3(0, self.box_size, 0)
        plane_left = Plane(
            anchor=left_anchor, normal=left_normal,
            u_dir=left_u_dir, v_dir=left_v_dir,
            u_len=self.box_size, v_len=self.box_size,
            material=materials['left']
        )
        scene.add_object(plane_left)

        # Right Wall: x = +15 (파란색)
        right_anchor = Vec3(half_size, -half_size, -half_size)
        right_normal = Vec3(-1, 0, 0)
        right_u_dir = Vec3(0, 0, self.box_size)
        right_v_dir = Vec3(0, self.box_size, 0)
        plane_right = Plane(
            anchor=right_anchor, normal=right_normal,
            u_dir=right_u_dir, v_dir=right_v_dir,
            u_len=self.box_size, v_len=self.box_size,
            material=materials['right']
        )
        scene.add_object(plane_right)

        # Ceiling: y = +15
        ceil_anchor = Vec3(-half_size, half_size, -half_size)
        ceil_normal = Vec3(0, -1, 0)
        ceil_u_dir = Vec3(self.box_size, 0, 0)
        ceil_v_dir = Vec3(0, 0, self.box_size)
        plane_ceil = Plane(
            anchor=ceil_anchor, normal=ceil_normal,
            u_dir=ceil_u_dir, v_dir=ceil_v_dir,
            u_len=self.box_size, v_len=self.box_size,
            material=materials['ceiling']
        )
        scene.add_object(plane_ceil)

        # # Ceiling: y = +15
        # ceil_anchor = Vec3(-half_size, half_size, -half_size)
        # ceil_normal = Vec3(0, -1, 0)
        # ceil_u_dir = Vec3(self.box_size, 0, 0)
        # ceil_v_dir = Vec3(0, 0, self.box_size)
        # plane_lightceil = Plane(
        #     anchor=ceil_anchor, normal=ceil_normal,
        #     u_dir=ceil_u_dir, v_dir=ceil_v_dir,
        #     u_len=4, v_len=4,
        #     material=materials['ceiling']
        # )
        # scene.add_object(plane_lightceil)

    def _create_rubiks_cubes(self, scene: Scene, materials: dict):
        """루빅스 큐브 2개 생성"""
        cube_half = self.cube_size / 2.0  # 2.8cm
        floor_y = -self.box_size / 2.0  # -15cm

        # 첫 번째 큐브: 바닥 정중앙, 225도 회전
        cube1_center = Vec3(0, floor_y + cube_half, 0)
        self._create_single_cube(scene, materials, cube1_center, rotation_y=225.0)

        # 두 번째 큐브: 첫 번째 큐브 바로 위, 정면
        cube2_center = Vec3(0, floor_y + cube_half + self.cube_size, 0)
        self._create_single_cube(scene, materials, cube2_center, rotation_y=0.0)

    def _create_single_cube(self, scene: Scene, materials: dict, center: Vec3, rotation_y: float):
        """단일 루빅스 큐브 생성"""
        half_size = self.cube_size / 2.0

        # 로컬 정점들
        local_verts = [
            Vec3(-half_size, -half_size, half_size),  # 0
            Vec3(half_size, -half_size, half_size),  # 1
            Vec3(half_size, half_size, half_size),  # 2
            Vec3(-half_size, half_size, half_size),  # 3
            Vec3(-half_size, -half_size, -half_size),  # 4
            Vec3(half_size, -half_size, -half_size),  # 5
            Vec3(half_size, half_size, -half_size),  # 6
            Vec3(-half_size, half_size, -half_size),  # 7
        ]

        def rotate_y(pt: Vec3, angle_deg: float):
            angle = math.radians(angle_deg)
            c = math.cos(angle)
            s = math.sin(angle)
            x = pt.x * c - pt.z * s
            z = pt.x * s + pt.z * c
            return Vec3(x, pt.y, z)

        # 월드 좌표로 변환
        world_verts = [center + rotate_y(v, rotation_y) for v in local_verts]

        # UV 좌표
        uv0 = np.array([0, 0])
        uv1 = np.array([1, 0])
        uv2 = np.array([1, 1])
        uv3 = np.array([0, 1])

        # 각 면의 정점 인덱스와 재질 (루빅스 큐브 색상)
        faces = [
            ((0, 1, 2, 3), materials['cube_red']),  # Front (+Z) - 흰색
            ((1, 5, 6, 2), materials['cube_blue']),  # Right (+X) - 빨강
            ((3, 2, 6, 7), materials['cube_yellow']),  # Top (+Y) - 노랑
            ((4, 5, 1, 0), materials['cube_white']),  # Bottom (-Y) - 주황
            ((4, 0, 3, 7), materials['cube_orange']),  # Left (-X) - 초록
            ((5, 4, 7, 6), materials['cube_green'])  # Back (-Z) - 파랑
        ]

        for (i0, i1, i2, i3), mat in faces:
            # 각 면을 두 개의 삼각형으로 분할
            scene.add_object(Triangle(
                world_verts[i0], world_verts[i1], world_verts[i2],
                uv0, uv1, uv2, mat
            ))
            scene.add_object(Triangle(
                world_verts[i0], world_verts[i2], world_verts[i3],
                uv0, uv2, uv3, mat
            ))

    def _create_spheres(self, scene: Scene, materials: dict):
        """구체들 생성 - 굴절 구체 포함"""
        floor_y = -self.box_size / 2.0  # -15cm

        # 유리구체 1 (바닥의 오른쪽)
        ball_radius = 3
        sphere_center = Vec3(self.box_size / 4, floor_y + ball_radius, self.box_size / 4)
        sphere = Sphere(center=sphere_center, radius=ball_radius, material=materials['glass'])
        scene.add_object(sphere)





        # 기존 금속 구체 (왼쪽 앞)
        metal_radius = 3
        metal_center = Vec3(-self.box_size / 4, floor_y + metal_radius, self.box_size / 4)
        metal_sphere = Sphere(center=metal_center, radius=metal_radius, material=materials['sphere_metal'])
        scene.add_object(metal_sphere)

        # ========== 새로 추가: 큐브 윗면에 유리 구체 ==========

        # 큐브 위치 계산
        cube1_center_y = floor_y + self.cube_size / 2.0  # 첫 번째 큐브 중심
        cube2_center_y = floor_y + self.cube_size / 2.0 + self.cube_size  # 두 번째 큐브 중심
        cube2_top_y = cube2_center_y + self.cube_size / 2.0  # 두 번째 큐브 윗면

        # 유리 구체 설정
        glass_radius = 3.0
        glass_center = Vec3(
            0,  # X: 큐브와 같은 중심축 (0)
            cube2_top_y + glass_radius,  # Y: 큐브 윗면 + 구체 반지름 (구체가 큐브 위에 올려짐)
            0  # Z: 큐브와 같은 중심축 (0)
        )

        glass_sphere = Sphere(
            center=glass_center,
            radius=glass_radius,
            material=materials['glass']
        )
        scene.add_object(glass_sphere)

        # ========== 굴절 구체들 추가 ==========

        # # 유리 구체 (중앙 뒤쪽)
        # glass_radius = 3.5
        # glass_center = Vec3(0, floor_y + glass_radius, -self.box_size / 4)
        # glass_sphere = Sphere(center=glass_center, radius=glass_radius, material=materials['glass'])
        # scene.add_object(glass_sphere)
        #
        # # 작은 크리스탈 구체 (오른쪽 뒤)
        # crystal_radius = 5
        # crystal_center = Vec3(self.box_size / 3, floor_y + crystal_radius, -self.box_size / 30)
        # crystal_sphere = Sphere(center=crystal_center, radius=crystal_radius, material=materials['crystal'])
        # scene.add_object(crystal_sphere)
        #
        # # 물방울 같은 구체 (왼쪽 뒤, 작고 높은 위치)
        # water_radius = 4.0
        # water_center = Vec3(-self.box_size / 3, floor_y + water_radius + 8, -self.box_size / 4)
        # water_sphere = Sphere(center=water_center, radius=water_radius, material=materials['water_sphere'])
        # scene.add_object(water_sphere)

    def _create_canvas(self, scene: Scene, materials: dict):
        """27.5x22x1.5cm 캔버스를 68도 각도로 배치 - 뒷벽에 기대어 있도록 수정"""
        back_wall_z = -self.box_size / 2.0  # -15cm (뒷벽 위치)
        floor_y = -self.box_size / 2.0  # -15cm (바닥 위치)

        # 캔버스 하단은 바닥에서 살짝 떨어뜨림
        canvas_bottom_y = floor_y + 0.5

        # 68도 각도 계산
        angle_rad = math.radians(self.canvas_angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        # 캔버스의 4개 모서리 계산
        half_width = self.canvas_width / 2.0

        # 캔버스가 뒷벽에 기대어 있으므로, 하단과 상단의 Z 좌표를 다르게 계산
        # 하단: 뒷벽에서 캔버스 두께만큼 앞으로
        bottom_z = back_wall_z + 6.5 * self.canvas_depth

        # 상단: 68도 각도로 기울어져서 더 앞으로 나옴
        top_z = bottom_z + self.canvas_height * cos_angle

        # 상단 Y 좌표: 68도 각도로 올라감
        top_y = canvas_bottom_y + self.canvas_height * sin_angle

        # 4개 모서리 정점
        bottom_left = Vec3(-half_width, canvas_bottom_y, bottom_z)
        bottom_right = Vec3(half_width, canvas_bottom_y, bottom_z)
        top_left = Vec3(-half_width, top_y, top_z)
        top_right = Vec3(half_width, top_y, top_z)

        # UV 좌표
        uv_bl = np.array([0, 0])  # bottom-left
        uv_br = np.array([1, 0])  # bottom-right
        uv_tl = np.array([0, 1])  # top-left
        uv_tr = np.array([1, 1])  # top-right

        # 캔버스를 두 개의 삼각형으로 분할
        scene.add_object(Triangle(
            bottom_left, bottom_right, top_right,
            uv_bl, uv_br, uv_tr, materials['canvas']
        ))
        scene.add_object(Triangle(
            bottom_left, top_right, top_left,
            uv_bl, uv_tr, uv_tl, materials['canvas']
        ))

    def _create_lighting(self, scene: Scene):
        """천장 중앙에 4x4cm 면광원 생성"""
        # 천장 중앙, 약간 아래쪽에 배치
        light_center = Vec3(0, self.box_size / 2 - 1, 0)
        u_vec = Vec3(1, 0, 0)
        v_vec = Vec3(0, 0, 1)

        create_area_light(
            scene, center=light_center,
            u_vec=u_vec, v_vec=v_vec,
            u_size=self.light_size, v_size=self.light_size,
            n_u=4, n_v=4  # 4x4에서 6x6으로 증가 (더 부드럽고 밝은 조명)
        )