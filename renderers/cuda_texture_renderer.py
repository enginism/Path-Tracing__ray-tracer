import math
import time
import numpy as np
from typing import List
from PIL import Image
from numba import cuda
import random

from core.math import Vec3, Ray
from core.material import HitRecord
from core.scene import Scene, RenderSettings
from core.camera import Camera
from core.geometry import Sphere, Plane, Triangle
from renderers.base_renderer import BaseRenderer, RendererFactory


@cuda.jit
def cuda_trace_kernel(output, scene_data, camera_data, light_data, texture_data, texture_info, width, height,
                      samples_per_pixel, max_depth):
    """CUDA 커널: 각 픽셀별로 레이트레이싱 수행 - 모든 객체 타입 + 굴절 지원"""
    # 스레드 인덱스 계산
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if x >= width or y >= height:
        return

    # 픽셀 인덱스
    pixel_idx = y * width + x

    # 랜덤 시드 (스레드별로 다르게)
    rng_state = x + y * width + 1

    # 색상 누적
    color_r = 0.0
    color_g = 0.0
    color_b = 0.0

    # 샘플링 (안티앨리어싱)
    grid_n = int(math.sqrt(samples_per_pixel))

    for a in range(grid_n):
        for b in range(grid_n):
            # Jittered 샘플링
            du = (a + cuda_random(rng_state)) / grid_n
            dv = (b + cuda_random(rng_state)) / grid_n
            rng_state = (rng_state * 1103515245 + 12345) & 0x7fffffff

            u = (x + du) / width
            v = (y + dv) / height

            # 레이 생성
            ray_origin, ray_direction = cuda_get_ray(camera_data, u, v)

            # 레이트레이싱
            r, g, b = cuda_trace_ray(scene_data, light_data, texture_data, texture_info,
                                     ray_origin, ray_direction, max_depth, rng_state)

            color_r += r
            color_g += g
            color_b += b

            rng_state = (rng_state * 1103515245 + 12345) & 0x7fffffff

    # 평균 계산
    color_r /= samples_per_pixel
    color_g /= samples_per_pixel
    color_b /= samples_per_pixel

    # 결과 저장 (gamma 보정 및 클램핑)
    output[pixel_idx * 3 + 0] = min(255, max(0, int(color_r * 255)))
    output[pixel_idx * 3 + 1] = min(255, max(0, int(color_g * 255)))
    output[pixel_idx * 3 + 2] = min(255, max(0, int(color_b * 255)))


@cuda.jit(device=True)
def cuda_random(state):
    """간단한 GPU 랜덤 함수"""
    state = (state * 1103515245 + 12345) & 0x7fffffff
    return (state / 2147483647.0)


@cuda.jit(device=True)
def cuda_get_ray(camera_data, u, v):
    """카메라에서 레이 생성"""
    origin_x = camera_data[0]
    origin_y = camera_data[1]
    origin_z = camera_data[2]

    llc_x = camera_data[3]
    llc_y = camera_data[4]
    llc_z = camera_data[5]

    h_x = camera_data[6]
    h_y = camera_data[7]
    h_z = camera_data[8]

    v_x = camera_data[9]
    v_y = camera_data[10]
    v_z = camera_data[11]

    # 방향 계산
    dir_x = llc_x + u * h_x + v * v_x - origin_x
    dir_y = llc_y + u * h_y + v * v_y - origin_y
    dir_z = llc_z + u * h_z + v * v_z - origin_z

    # 정규화
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
    if length > 0:
        dir_x /= length
        dir_y /= length
        dir_z /= length

    return (origin_x, origin_y, origin_z), (dir_x, dir_y, dir_z)


@cuda.jit(device=True)
def cuda_sample_texture(texture_data, texture_start, texture_width, texture_height, u, v):
    """GPU에서 텍스처 샘플링"""
    # UV 좌표 클램핑
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))

    # UV 좌표를 이미지 좌표로 변환
    iu = int(u * (texture_width - 1))
    iv = int((1.0 - v) * (texture_height - 1))  # V 뒤집기

    # 경계 체크
    iu = max(0, min(texture_width - 1, iu))
    iv = max(0, min(texture_height - 1, iv))

    # 텍스처 인덱스 계산
    pixel_idx = iv * texture_width + iu
    base_idx = texture_start + pixel_idx * 3

    # RGB 값 반환 (0-255 → 0-1)
    if base_idx + 2 < len(texture_data):
        r = texture_data[base_idx + 0] / 255.0
        g = texture_data[base_idx + 1] / 255.0
        b = texture_data[base_idx + 2] / 255.0
        return r, g, b
    else:
        return 1.0, 1.0, 1.0  # 기본 흰색


@cuda.jit(device=True)
def cuda_refract(incident_x, incident_y, incident_z, normal_x, normal_y, normal_z, ni_over_nt):
    """GPU에서 굴절 벡터 계산 (스넬의 법칙)"""
    # 입사벡터와 법선의 내적
    cos_i = -(incident_x * normal_x + incident_y * normal_y + incident_z * normal_z)

    # 전반사 판별: sin²(theta_t) = (ni/nt)² * (1 - cos²(theta_i))
    sin2_t = ni_over_nt * ni_over_nt * (1.0 - cos_i * cos_i)

    if sin2_t > 1.0:
        # 전반사 발생
        return False, 0.0, 0.0, 0.0

    # 굴절 벡터 계산
    cos_t = math.sqrt(1.0 - sin2_t)

    # R = ni/nt * I + (ni/nt * cos_i - cos_t) * N
    factor1 = ni_over_nt
    factor2 = ni_over_nt * cos_i - cos_t

    refracted_x = factor1 * incident_x + factor2 * normal_x
    refracted_y = factor1 * incident_y + factor2 * normal_y
    refracted_z = factor1 * incident_z + factor2 * normal_z

    return True, refracted_x, refracted_y, refracted_z


@cuda.jit(device=True)
def cuda_trace_ray(scene_data, light_data, texture_data, texture_info, ray_origin, ray_direction, max_depth, rng_state):
    """GPU에서 레이트레이싱 수행 - 반사 + 굴절 지원"""
    color_r = 0.0
    color_g = 0.0
    color_b = 0.0

    current_origin = ray_origin
    current_direction = ray_direction
    attenuation_r = 1.0
    attenuation_g = 1.0
    attenuation_b = 1.0

    for depth in range(max_depth):
        # 씬과의 교차 검사
        hit, hit_t, hit_point, hit_normal, material_data, uv_coords = cuda_scene_hit(
            scene_data, current_origin, current_direction, 0.001, 1000000.0
        )

        if hit:
            # 재질 정보 분해
            mat_color_r = material_data[0]
            mat_color_g = material_data[1]
            mat_color_b = material_data[2]
            mat_diffuse = material_data[3]
            mat_specular = material_data[4]
            mat_reflective = material_data[5]
            mat_refractive = material_data[6]  # 굴절 추가
            mat_ior = material_data[7]  # 굴절률 추가
            has_texture = material_data[8] > 0.5
            texture_id = int(material_data[9])

            # 텍스처 샘플링
            if has_texture and texture_id >= 0 and texture_id < len(texture_info) // 3:
                tex_start = int(texture_info[texture_id * 3 + 0])
                tex_width = int(texture_info[texture_id * 3 + 1])
                tex_height = int(texture_info[texture_id * 3 + 2])

                # 텍스처에서 색상 샘플링
                tex_r, tex_g, tex_b = cuda_sample_texture(
                    texture_data, tex_start, tex_width, tex_height, uv_coords[0], uv_coords[1]
                )

                # 베이스 색상과 텍스처 색상 결합
                mat_color_r = tex_r
                mat_color_g = tex_g
                mat_color_b = tex_b

            # 기본 ambient 조명
            ambient_factor = 0.4
            local_color_r = mat_color_r * ambient_factor
            local_color_g = mat_color_g * ambient_factor
            local_color_b = mat_color_b * ambient_factor

            # Area Light 조명 계산
            num_lights = int(light_data[0])
            if num_lights > 0:
                diffuse_contribution_r = 0.0
                diffuse_contribution_g = 0.0
                diffuse_contribution_b = 0.0

                specular_contribution_r = 0.0
                specular_contribution_g = 0.0
                specular_contribution_b = 0.0

                for i in range(num_lights):
                    light_offset = 1 + i * 3
                    light_x = light_data[light_offset + 0]
                    light_y = light_data[light_offset + 1]
                    light_z = light_data[light_offset + 2]

                    # 조명 방향과 거리
                    light_dx = light_x - hit_point[0]
                    light_dy = light_y - hit_point[1]
                    light_dz = light_z - hit_point[2]

                    light_dist = math.sqrt(light_dx * light_dx + light_dy * light_dy + light_dz * light_dz)
                    if light_dist > 0.001:
                        light_dx /= light_dist
                        light_dy /= light_dist
                        light_dz /= light_dist

                        # 그림자 레이 생성
                        shadow_origin = (
                            hit_point[0] + hit_normal[0] * 0.001,
                            hit_point[1] + hit_normal[1] * 0.001,
                            hit_point[2] + hit_normal[2] * 0.001
                        )
                        shadow_direction = (light_dx, light_dy, light_dz)

                        # 그림자 체크
                        shadow_hit, shadow_t, _, _, _, _ = cuda_scene_hit(
                            scene_data, shadow_origin, shadow_direction, 0.001, light_dist - 0.001
                        )

                        if not shadow_hit:
                            # Diffuse (Lambert) 조명
                            diff_factor = max(0.0,
                                              hit_normal[0] * light_dx +
                                              hit_normal[1] * light_dy +
                                              hit_normal[2] * light_dz
                                              )

                            # 거리에 따른 감쇠 2 -> 1.5
                            attenuation = 1.5 / (1.0 + 0.001 * light_dist + 0.0001 * light_dist * light_dist)
                            diffuse_intensity = diff_factor * attenuation / num_lights

                            # 숫자 곱해서 직접조명 컨트롤 / 밝기
                            diffuse_contribution_r += mat_color_r * diffuse_intensity * mat_diffuse * 0.6
                            diffuse_contribution_g += mat_color_g * diffuse_intensity * mat_diffuse * 0.6
                            diffuse_contribution_b += mat_color_b * diffuse_intensity * mat_diffuse * 0.6

                            # === SPECULAR (Phong) 조명 추가 ===
                            if mat_specular > 0.01 and diff_factor > 0.0:
                                # 반사 벡터 계산: R = 2(N·L)N - L
                                dot_nl = (hit_normal[0] * light_dx +
                                          hit_normal[1] * light_dy +
                                          hit_normal[2] * light_dz)

                                reflect_x = 2.0 * dot_nl * hit_normal[0] - light_dx
                                reflect_y = 2.0 * dot_nl * hit_normal[1] - light_dy
                                reflect_z = 2.0 * dot_nl * hit_normal[2] - light_dz

                                # 시선 벡터 (카메라 방향)
                                view_x = -current_direction[0]
                                view_y = -current_direction[1]
                                view_z = -current_direction[2]

                                # 스페큘러 계수: (R·V)^shininess
                                dot_rv = max(0.0, reflect_x * view_x + reflect_y * view_y + reflect_z * view_z)

                                # Shininess 값 (재질에 따라 다름)
                                shininess = 32.0  # 기본값: 32 (플라스틱)
                                specular_multiplier = 1.0

                                if mat_reflective > 0.9 and mat_specular > 0.9:  # 크롬 금속
                                    shininess = 256.0  # 매우 날카로운 하이라이트
                                    specular_multiplier = 1.5  # 하이라이트 강도 50% 증가
                                elif mat_reflective > 0.7:  # 일반 금속
                                    shininess = 128.0  # 날카로운 하이라이트
                                    specular_multiplier = 1.2  # 하이라이트 강도 20% 증가
                                elif mat_specular > 0.5:  # 매우 광택
                                    shininess = 64.0

                                spec_factor = pow(dot_rv, shininess)
                                specular_intensity = spec_factor * attenuation * specular_multiplier / num_lights

                                # 금속은 색상이 있는 하이라이트 (베이스 색상 반영)
                                if mat_reflective > 0.7:
                                    # 금속성 하이라이트 - 재질 색상 반영
                                    specular_contribution_r += specular_intensity * mat_specular * mat_color_r
                                    specular_contribution_g += specular_intensity * mat_specular * mat_color_g
                                    specular_contribution_b += specular_intensity * mat_specular * mat_color_b
                                else:
                                    # 일반 스페큘러 - 흰색 하이라이트
                                    specular_contribution_r += specular_intensity * mat_specular
                                    specular_contribution_g += specular_intensity * mat_specular
                                    specular_contribution_b += specular_intensity * mat_specular

                local_color_r += diffuse_contribution_r + specular_contribution_r
                local_color_g += diffuse_contribution_g + specular_contribution_g
                local_color_b += diffuse_contribution_b + specular_contribution_b

            # 현재 깊이에서의 기여도 추가 수정
            # 투명 재질도 약간의 직접 조명을 받도록 함
            base_contribution = max(0.1, 1.0 - mat_reflective - mat_refractive)  # 최소 0.1 보장
            color_r += local_color_r * attenuation_r * base_contribution
            color_g += local_color_g * attenuation_g * base_contribution
            color_b += local_color_b * attenuation_b * base_contribution

            # 반사와 굴절 처리 수정
            if (mat_reflective > 0.01 or mat_refractive > 0.01) and depth < max_depth - 1:
                # Fresnel 효과 간소화: 굴절을 우선하되 일부 반사도 처리
                use_refraction = mat_refractive > mat_reflective and mat_refractive > 0.1

                if use_refraction:
                    # 굴절 처리 개선
                    dot_nd = (current_direction[0] * hit_normal[0] +
                              current_direction[1] * hit_normal[1] +
                              current_direction[2] * hit_normal[2])

                    if dot_nd > 0:
                        # 물체 내부에서 외부로
                        outward_normal = (-hit_normal[0], -hit_normal[1], -hit_normal[2])
                        ni_over_nt = mat_ior
                        offset_dir = hit_normal  # 외부로 나갈 때는 법선 방향
                    else:
                        # 외부에서 물체 내부로
                        outward_normal = hit_normal
                        ni_over_nt = 1.0 / mat_ior
                        offset_dir = (-hit_normal[0], -hit_normal[1], -hit_normal[2])  # 내부로 들어갈 때는 법선 반대

                    # 굴절 계산
                    refracted, refract_x, refract_y, refract_z = cuda_refract(
                        current_direction[0], current_direction[1], current_direction[2],
                        outward_normal[0], outward_normal[1], outward_normal[2],
                        ni_over_nt
                    )

                    if refracted:
                        # 굴절 성공 - 올바른 방향으로 오프셋
                        current_origin = (
                            hit_point[0] + offset_dir[0] * 0.001,
                            hit_point[1] + offset_dir[1] * 0.001,
                            hit_point[2] + offset_dir[2] * 0.001
                        )
                        current_direction = (refract_x, refract_y, refract_z)

                        attenuation_r *= mat_refractive * 0.95  # 약간의 감쇠
                        attenuation_g *= mat_refractive * 0.95
                        attenuation_b *= mat_refractive * 0.95
                    else:
                        # 전반사 → 반사로 처리
                        dot_product = (current_direction[0] * hit_normal[0] +
                                       current_direction[1] * hit_normal[1] +
                                       current_direction[2] * hit_normal[2])

                        reflect_x = current_direction[0] - 2.0 * dot_product * hit_normal[0]
                        reflect_y = current_direction[1] - 2.0 * dot_product * hit_normal[1]
                        reflect_z = current_direction[2] - 2.0 * dot_product * hit_normal[2]

                        current_origin = (
                            hit_point[0] + hit_normal[0] * 0.001,
                            hit_point[1] + hit_normal[1] * 0.001,
                            hit_point[2] + hit_normal[2] * 0.001
                        )
                        current_direction = (reflect_x, reflect_y, reflect_z)

                        attenuation_r *= mat_reflective
                        attenuation_g *= mat_reflective
                        attenuation_b *= mat_reflective
                else:
                    # 반사 처리 (기존과 동일)
                    dot_product = (current_direction[0] * hit_normal[0] +
                                   current_direction[1] * hit_normal[1] +
                                   current_direction[2] * hit_normal[2])

                    reflect_x = current_direction[0] - 2.0 * dot_product * hit_normal[0]
                    reflect_y = current_direction[1] - 2.0 * dot_product * hit_normal[1]
                    reflect_z = current_direction[2] - 2.0 * dot_product * hit_normal[2]

                    current_origin = (
                        hit_point[0] + hit_normal[0] * 0.001,
                        hit_point[1] + hit_normal[1] * 0.001,
                        hit_point[2] + hit_normal[2] * 0.001
                    )
                    current_direction = (reflect_x, reflect_y, reflect_z)

                    attenuation_r *= mat_reflective
                    attenuation_g *= mat_reflective
                    attenuation_b *= mat_reflective
            else:
                break
        else:
            # 배경색 (검은색)
            break

    return color_r, color_g, color_b


@cuda.jit(device=True)
def cuda_scene_hit(scene_data, ray_origin, ray_direction, t_min, t_max):
    """씬과의 교차 검사 - Plane, Sphere, Triangle 모두 지원"""
    closest_t = t_max
    hit = False
    hit_point = (0.0, 0.0, 0.0)
    hit_normal = (0.0, 1.0, 0.0)
    material_data = (0.5, 0.5, 0.5, 0.8, 0.2, 0.0, 0.0, 1.0, 0.0, -1.0)
    uv_coords = (0.0, 0.0)

    offset = 0

    # ======== Planes 처리 ========
    num_planes = int(scene_data[offset])
    offset += 1

    for i in range(num_planes):
        plane_offset = offset + i * 20  # 기본 Plane 데이터 (굴절 정보 제외)

        anchor_x = scene_data[plane_offset + 0]
        anchor_y = scene_data[plane_offset + 1]
        anchor_z = scene_data[plane_offset + 2]

        normal_x = scene_data[plane_offset + 3]
        normal_y = scene_data[plane_offset + 4]
        normal_z = scene_data[plane_offset + 5]

        u_x = scene_data[plane_offset + 6]
        u_y = scene_data[plane_offset + 7]
        u_z = scene_data[plane_offset + 8]

        v_x = scene_data[plane_offset + 9]
        v_y = scene_data[plane_offset + 10]
        v_z = scene_data[plane_offset + 11]

        u_len = scene_data[plane_offset + 12]
        v_len = scene_data[plane_offset + 13]

        color_r = scene_data[plane_offset + 14]
        color_g = scene_data[plane_offset + 15]
        color_b = scene_data[plane_offset + 16]

        mat_diffuse = scene_data[plane_offset + 17]
        mat_specular = scene_data[plane_offset + 18]
        mat_reflective = scene_data[plane_offset + 19]

        # 레이-평면 교차 검사
        denom = normal_x * ray_direction[0] + normal_y * ray_direction[1] + normal_z * ray_direction[2]

        if abs(denom) > 1e-6:
            diff_x = anchor_x - ray_origin[0]
            diff_y = anchor_y - ray_origin[1]
            diff_z = anchor_z - ray_origin[2]

            t = (diff_x * normal_x + diff_y * normal_y + diff_z * normal_z) / denom

            if t_min < t < closest_t:
                hit_x = ray_origin[0] + t * ray_direction[0]
                hit_y = ray_origin[1] + t * ray_direction[1]
                hit_z = ray_origin[2] + t * ray_direction[2]

                v2_x = hit_x - anchor_x
                v2_y = hit_y - anchor_y
                v2_z = hit_z - anchor_z

                u_unit_len = math.sqrt(u_x * u_x + u_y * u_y + u_z * u_z)
                v_unit_len = math.sqrt(v_x * v_x + v_y * v_y + v_z * v_z)

                if u_unit_len > 0 and v_unit_len > 0:
                    u_unit_x = u_x / u_unit_len
                    u_unit_y = u_y / u_unit_len
                    u_unit_z = u_z / u_unit_len

                    v_unit_x = v_x / v_unit_len
                    v_unit_y = v_y / v_unit_len
                    v_unit_z = v_z / v_unit_len

                    u_hit = v2_x * u_unit_x + v2_y * u_unit_y + v2_z * u_unit_z
                    v_hit = v2_x * v_unit_x + v2_y * v_unit_y + v2_z * v_unit_z

                    if 0 <= u_hit <= u_len and 0 <= v_hit <= v_len:
                        closest_t = t
                        hit = True
                        hit_point = (hit_x, hit_y, hit_z)
                        hit_normal = (normal_x, normal_y, normal_z)
                        uv_coords = (u_hit / u_len, v_hit / v_len)
                        material_data = (color_r, color_g, color_b, mat_diffuse, mat_specular,
                                         mat_reflective, 0.0, 1.0, 0.0, -1.0)

    offset += num_planes * 20

    # ======== Spheres 처리 ========
    num_spheres = int(scene_data[offset])
    offset += 1

    for i in range(num_spheres):
        sphere_offset = offset + i * 12  # 기본 Sphere 데이터 (굴절 정보 제외)

        center_x = scene_data[sphere_offset + 0]
        center_y = scene_data[sphere_offset + 1]
        center_z = scene_data[sphere_offset + 2]
        radius = scene_data[sphere_offset + 3]

        color_r = scene_data[sphere_offset + 4]
        color_g = scene_data[sphere_offset + 5]
        color_b = scene_data[sphere_offset + 6]

        mat_diffuse = scene_data[sphere_offset + 7]
        mat_specular = scene_data[sphere_offset + 8]
        mat_reflective = scene_data[sphere_offset + 9]

        # ✅ 굴절 정보 읽기 추가
        mat_refractive = scene_data[sphere_offset + 10]
        mat_ior = scene_data[sphere_offset + 11]

        # 레이-구체 교차 검사
        oc_x = ray_origin[0] - center_x
        oc_y = ray_origin[1] - center_y
        oc_z = ray_origin[2] - center_z

        a = (ray_direction[0] * ray_direction[0] +
             ray_direction[1] * ray_direction[1] +
             ray_direction[2] * ray_direction[2])

        b = (oc_x * ray_direction[0] +
             oc_y * ray_direction[1] +
             oc_z * ray_direction[2])

        c = (oc_x * oc_x + oc_y * oc_y + oc_z * oc_z) - radius * radius

        discriminant = b * b - a * c

        if discriminant > 0:
            sqrt_d = math.sqrt(discriminant)
            t1 = (-b - sqrt_d) / a
            t2 = (-b + sqrt_d) / a

            t = t1 if t_min < t1 < closest_t else (t2 if t_min < t2 < closest_t else -1)

            if t > 0:
                closest_t = t
                hit = True

                hit_x = ray_origin[0] + t * ray_direction[0]
                hit_y = ray_origin[1] + t * ray_direction[1]
                hit_z = ray_origin[2] + t * ray_direction[2]

                normal_x = (hit_x - center_x) / radius
                normal_y = (hit_y - center_y) / radius
                normal_z = (hit_z - center_z) / radius

                hit_point = (hit_x, hit_y, hit_z)
                hit_normal = (normal_x, normal_y, normal_z)
                uv_coords = (0.0, 0.0)  # 구체 UV 매핑은 간단히 생략

                # ✅ 굴절 정보를 포함한 material_data
                material_data = (color_r, color_g, color_b, mat_diffuse, mat_specular,
                                 mat_reflective, mat_refractive, mat_ior, 0.0, -1.0)

    offset += num_spheres * 12

    # ======== Triangles 처리 ========
    num_triangles = int(scene_data[offset])
    offset += 1

    for i in range(num_triangles):
        tri_offset = offset + i * 26  # 26개 값으로 수정

        v0_x = scene_data[tri_offset + 0]
        v0_y = scene_data[tri_offset + 1]
        v0_z = scene_data[tri_offset + 2]

        v1_x = scene_data[tri_offset + 3]
        v1_y = scene_data[tri_offset + 4]
        v1_z = scene_data[tri_offset + 5]

        v2_x = scene_data[tri_offset + 6]
        v2_y = scene_data[tri_offset + 7]
        v2_z = scene_data[tri_offset + 8]

        normal_x = scene_data[tri_offset + 9]
        normal_y = scene_data[tri_offset + 10]
        normal_z = scene_data[tri_offset + 11]

        color_r = scene_data[tri_offset + 12]
        color_g = scene_data[tri_offset + 13]
        color_b = scene_data[tri_offset + 14]

        mat_diffuse = scene_data[tri_offset + 15]
        mat_specular = scene_data[tri_offset + 16]
        mat_reflective = scene_data[tri_offset + 17]

        has_texture = scene_data[tri_offset + 18]
        texture_id = scene_data[tri_offset + 19]

        # UV 좌표들
        uv0_u = scene_data[tri_offset + 20]
        uv0_v = scene_data[tri_offset + 21]
        uv1_u = scene_data[tri_offset + 22]
        uv1_v = scene_data[tri_offset + 23]
        uv2_u = scene_data[tri_offset + 24]
        uv2_v = scene_data[tri_offset + 25]

        # 레이-삼각형 교차 검사 (Möller-Trumbore 알고리즘)
        edge1_x = v1_x - v0_x
        edge1_y = v1_y - v0_y
        edge1_z = v1_z - v0_z

        edge2_x = v2_x - v0_x
        edge2_y = v2_y - v0_y
        edge2_z = v2_z - v0_z

        # h = ray_direction × edge2
        h_x = ray_direction[1] * edge2_z - ray_direction[2] * edge2_y
        h_y = ray_direction[2] * edge2_x - ray_direction[0] * edge2_z
        h_z = ray_direction[0] * edge2_y - ray_direction[1] * edge2_x

        # a = edge1 · h
        a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z

        if abs(a) < 1e-6:
            continue  # 레이가 삼각형과 평행

        f = 1.0 / a
        s_x = ray_origin[0] - v0_x
        s_y = ray_origin[1] - v0_y
        s_z = ray_origin[2] - v0_z

        u = f * (s_x * h_x + s_y * h_y + s_z * h_z)
        if u < 0.0 or u > 1.0:
            continue

        # q = s × edge1
        q_x = s_y * edge1_z - s_z * edge1_y
        q_y = s_z * edge1_x - s_x * edge1_z
        q_z = s_x * edge1_y - s_y * edge1_x

        v = f * (ray_direction[0] * q_x + ray_direction[1] * q_y + ray_direction[2] * q_z)
        if v < 0.0 or u + v > 1.0:
            continue

        # t 계산
        t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z)

        if t_min < t < closest_t:
            closest_t = t
            hit = True

            hit_x = ray_origin[0] + t * ray_direction[0]
            hit_y = ray_origin[1] + t * ray_direction[1]
            hit_z = ray_origin[2] + t * ray_direction[2]

            # 법선 방향 확인
            dot_product = normal_x * ray_direction[0] + normal_y * ray_direction[1] + normal_z * ray_direction[2]
            if dot_product > 0:
                normal_x = -normal_x
                normal_y = -normal_y
                normal_z = -normal_z

            hit_point = (hit_x, hit_y, hit_z)
            hit_normal = (normal_x, normal_y, normal_z)

            # 올바른 UV 좌표 계산 (barycentric interpolation)
            w = 1.0 - u - v  # barycentric 좌표 가중치
            final_u = w * uv0_u + u * uv1_u + v * uv2_u
            final_v = w * uv0_v + u * uv1_v + v * uv2_v
            uv_coords = (final_u, final_v)

            material_data = (color_r, color_g, color_b, mat_diffuse, mat_specular,
                             mat_reflective, 0.0, 1.0, has_texture, texture_id)

    return hit, closest_t, hit_point, hit_normal, material_data, uv_coords


class CUDATextureRenderer(BaseRenderer):
    """텍스처 지원 CUDA 가속 레이트레이싱 렌더러 (굴절 기능 포함)"""

    def __init__(self):
        super().__init__("cuda_texture_raytracer")
        self._check_cuda_available()

    def _check_cuda_available(self):
        """CUDA 사용 가능 여부 확인"""
        try:
            cuda.select_device(0)
            print(f"CUDA 디바이스 감지: {cuda.get_current_device().name.decode()}")
        except Exception as e:
            raise RuntimeError(f"CUDA를 사용할 수 없습니다: {e}")

    def get_capabilities(self) -> List[str]:
        return [
            "ray_tracing",
            "shadows",
            "reflection",
            "refraction",  # 굴절 추가
            "textures",
            "gpu_acceleration",
            "anti_aliasing",
            "cuda_compute",
            "all_geometry_types"  # 모든 기하학 타입 지원
        ]

    def render(self, scene: Scene, camera: Camera, settings: RenderSettings) -> Image.Image:
        """CUDA를 사용한 GPU 렌더링 - 모든 객체 타입 + 텍스처 지원"""
        start_time = time.time()

        print(f"CUDA 텍스처 렌더링 시작: {settings.width}x{settings.height}, {settings.samples_per_pixel} samples")

        # 씬 데이터와 텍스처 데이터 준비
        scene_data = self._prepare_scene_data(scene)
        camera_data = self._prepare_camera_data(camera)
        light_data = self._prepare_light_data(scene)
        texture_data, texture_info = self._prepare_texture_data(scene)

        print(f"씬 데이터 크기: {len(scene_data)} floats")
        print(f"텍스처 데이터 크기: {len(texture_data)} bytes")
        print(f"조명 데이터: {len(scene.lights)} 개")

        # GPU 메모리 할당
        output_size = settings.width * settings.height * 3
        d_output = cuda.device_array(output_size, dtype=np.uint8)
        d_scene_data = cuda.to_device(scene_data)
        d_camera_data = cuda.to_device(camera_data)
        d_light_data = cuda.to_device(light_data)
        d_texture_data = cuda.to_device(texture_data)
        d_texture_info = cuda.to_device(texture_info)

        # CUDA 커널 실행 설정
        threads_per_block = (16, 16)
        blocks_per_grid_x = (settings.width + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (settings.height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        print(f"CUDA 설정: {blocks_per_grid} blocks, {threads_per_block} threads/block")

        # 커널 실행
        cuda_trace_kernel[blocks_per_grid, threads_per_block](
            d_output, d_scene_data, d_camera_data, d_light_data, d_texture_data, d_texture_info,
            settings.width, settings.height,
            settings.samples_per_pixel, settings.max_depth
        )

        # GPU에서 결과 복사
        output_array = d_output.copy_to_host()

        # 이미지 생성
        image_array = output_array.reshape((settings.height, settings.width, 3))
        image_array = np.flip(image_array, axis=0)  # Y축 뒤집기

        end_time = time.time()
        elapsed = end_time - start_time
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"CUDA 텍스처 렌더링 완료: {minutes}분 {seconds:.2f}초")

        return Image.fromarray(image_array, 'RGB')

    def _prepare_scene_data(self, scene: Scene) -> np.ndarray:
        """씬 데이터를 GPU 친화적 형태로 변환 - 안정적인 텍스처 매핑"""
        data = []

        # 텍스처 매핑 생성 (순서 보장)
        texture_map = {}
        texture_paths = []

        # 먼저 모든 텍스처 경로를 순서대로 수집
        for obj in scene.objects:
            if hasattr(obj, 'material') and obj.material.texture is not None:
                texture_path = getattr(obj.material.texture, 'path', None)
                if texture_path and texture_path not in texture_paths:
                    texture_paths.append(texture_path)

        # 경로를 정렬해서 항상 같은 순서로 처리
        texture_paths.sort()

        # 텍스처 ID 할당 (항상 같은 순서)
        for i, texture_path in enumerate(texture_paths):
            texture_map[texture_path] = i

        print(f"텍스처 경로 (정렬됨): {texture_paths}")
        print(f"텍스처 매핑: {texture_map}")

        planes = []
        spheres = []
        triangles = []

        print("=== 씬 객체 분석 ===")
        for i, obj in enumerate(scene.objects):
            print(f"객체 {i}: {type(obj).__name__}")

            if isinstance(obj, Plane):
                print(f"  - Plane: anchor={obj.anchor}, normal={obj.normal}")
                print(f"  - 재질: diffuse={obj.material.diffuse}, reflective={obj.material.reflective}")
                mat = obj.material

                planes.extend([
                    obj.anchor.x, obj.anchor.y, obj.anchor.z,
                    obj.normal.x, obj.normal.y, obj.normal.z,
                    obj.u_dir.x, obj.u_dir.y, obj.u_dir.z,
                    obj.v_dir.x, obj.v_dir.y, obj.v_dir.z,
                    obj.u_len, obj.v_len,
                    mat.color.x, mat.color.y, mat.color.z,
                    mat.diffuse, mat.specular, mat.reflective
                ])

            elif isinstance(obj, Sphere):
                print(f"  - Sphere: center={obj.center}, radius={obj.radius}")
                print(f"  - 재질: diffuse={obj.material.diffuse}, reflective={obj.material.reflective}")
                if hasattr(obj.material, 'refractive'):
                    print(f"  - 굴절: refractive={obj.material.refractive}, ior={obj.material.ior}")
                mat = obj.material

                # 굴절 정보를 포함한 Sphere 데이터
                spheres.extend([
                    obj.center.x, obj.center.y, obj.center.z, obj.radius,
                    mat.color.x, mat.color.y, mat.color.z,
                    mat.diffuse, mat.specular, mat.reflective,
                    mat.refractive, mat.ior  # 굴절 정보 추가
                ])

            elif isinstance(obj, Triangle):
                print(f"  - Triangle: 재질={obj.material.color if obj.material else 'None'}")
                if obj.material:
                    print(f"    - 텍스처: {'있음' if obj.material.texture else '없음'}")
                mat = obj.material

                # 텍스처 정보 (안정적인 ID 할당)
                has_texture = 1.0 if mat.texture is not None else 0.0
                tex_id = -1.0
                if mat.texture is not None:
                    texture_path = getattr(mat.texture, 'path', None)
                    if texture_path in texture_map:
                        tex_id = float(texture_map[texture_path])
                        print(f"    - 텍스처 경로: {texture_path} → ID: {tex_id}")

                # UV 좌표 추가 (실제 정점 UV 사용)
                uv0_u = obj.uv0[0] if obj.uv0 is not None else 0.0
                uv0_v = obj.uv0[1] if obj.uv0 is not None else 0.0
                uv1_u = obj.uv1[0] if obj.uv1 is not None else 1.0
                uv1_v = obj.uv1[1] if obj.uv1 is not None else 0.0
                uv2_u = obj.uv2[0] if obj.uv2 is not None else 1.0
                uv2_v = obj.uv2[1] if obj.uv2 is not None else 1.0

                print(f"    - UV0: ({uv0_u:.2f}, {uv0_v:.2f})")
                print(f"    - UV1: ({uv1_u:.2f}, {uv1_v:.2f})")
                print(f"    - UV2: ({uv2_u:.2f}, {uv2_v:.2f})")

                triangles.extend([
                    obj.v0.x, obj.v0.y, obj.v0.z,  # 0,1,2
                    obj.v1.x, obj.v1.y, obj.v1.z,  # 3,4,5
                    obj.v2.x, obj.v2.y, obj.v2.z,  # 6,7,8
                    obj.normal.x, obj.normal.y, obj.normal.z,  # 9,10,11
                    mat.color.x, mat.color.y, mat.color.z,  # 12,13,14
                    mat.diffuse, mat.specular, mat.reflective,  # 15,16,17
                    has_texture, tex_id,  # 18,19
                    uv0_u, uv0_v, uv1_u, uv1_v, uv2_u, uv2_v  # 20,21,22,23,24,25
                ])

        # 데이터 합치기 (Sphere는 이제 12개 값)
        data.append(len(planes) // 20)  # plane 개수 (20개 값씩)
        data.extend(planes)

        data.append(len(spheres) // 12)  # sphere 개수 (12개 값씩) - 굴절 정보 포함
        data.extend(spheres)

        data.append(len(triangles) // 26)  # triangle 개수 (26개 값씩)
        data.extend(triangles)

        print(f"=== GPU 씬 데이터 ===")
        print(f"Planes: {len(planes) // 20}개")
        print(f"Spheres: {len(spheres) // 12}개")  # 12개로 수정
        print(f"Triangles: {len(triangles) // 26}개")
        print(f"총 데이터 크기: {len(data)} floats")
        print(f"최종 텍스처 매핑: {texture_map}")

        return np.array(data, dtype=np.float32)

    def _prepare_texture_data(self, scene: Scene):
        """텍스처 데이터를 GPU로 준비 - 안정적인 순서 보장"""
        texture_data = []
        texture_info = []
        current_offset = 0

        # 모든 재질에서 텍스처 수집 (순서 보장)
        texture_paths = []
        for obj in scene.objects:
            if hasattr(obj, 'material') and obj.material.texture is not None:
                texture_path = getattr(obj.material.texture, 'path', None)
                if texture_path and texture_path not in texture_paths:
                    texture_paths.append(texture_path)

        # 경로를 정렬해서 항상 같은 순서로 처리
        texture_paths.sort()

        print(f"텍스처 로딩 순서: {texture_paths}")

        for i, texture_path in enumerate(texture_paths):
            try:
                # 텍스처 로드
                img = Image.open(texture_path).convert("RGB")
                width, height = img.size
                pixels = np.array(img)

                # 1D 배열로 변환
                pixel_data = pixels.reshape(-1).astype(np.uint8)

                # 텍스처 정보 저장
                texture_info.extend([current_offset, width, height])

                # 텍스처 데이터 추가
                texture_data.extend(pixel_data.tolist())
                current_offset += len(pixel_data)

                print(f"텍스처 {i}: {texture_path} ({width}x{height}) - 오프셋: {texture_info[i * 3]}")

            except Exception as e:
                print(f"텍스처 로드 실패 {texture_path}: {e}")
                # 기본 텍스처 (1x1 흰색)
                texture_info.extend([current_offset, 1, 1])
                texture_data.extend([255, 255, 255])
                current_offset += 3

        return np.array(texture_data, dtype=np.uint8), np.array(texture_info, dtype=np.int32)

    def _prepare_camera_data(self, camera: Camera) -> np.ndarray:
        """카메라 데이터를 GPU 친화적 형태로 변환"""
        return np.array([
            camera.origin.x, camera.origin.y, camera.origin.z,
            camera.lower_left_corner.x, camera.lower_left_corner.y, camera.lower_left_corner.z,
            camera.horizontal.x, camera.horizontal.y, camera.horizontal.z,
            camera.vertical.x, camera.vertical.y, camera.vertical.z
        ], dtype=np.float32)

    def _prepare_light_data(self, scene: Scene) -> np.ndarray:
        """조명 데이터를 GPU 친화적 형태로 변환"""
        data = [len(scene.lights)]

        for light_pos in scene.lights:
            data.extend([light_pos.x, light_pos.y, light_pos.z])

        return np.array(data, dtype=np.float32)


# 렌더러 등록
RendererFactory.register("cuda_texture_raytracer", CUDATextureRenderer)