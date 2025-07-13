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
def cuda_path_trace_kernel(output, scene_data, camera_data, light_data, texture_data, texture_info,
                           width, height, samples_per_pixel, max_depth, frame_count):
    """CUDA Path Tracing 커널 - Global Illumination 지원"""
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if x >= width or y >= height:
        return

    pixel_idx = y * width + x
    rng_state = (x + y * width + frame_count * width * height) * 1103515245 + 12345

    color_r = 0.0
    color_g = 0.0
    color_b = 0.0

    for sample in range(samples_per_pixel):
        u = (x + cuda_random(rng_state)) / width
        v = (y + cuda_random(rng_state)) / height
        rng_state = cuda_xorshift(rng_state)

        ray_origin, ray_direction = cuda_get_ray(camera_data, u, v)
        r, g, b = cuda_trace_path(scene_data, light_data, texture_data, texture_info,
                                  ray_origin, ray_direction, max_depth, rng_state)

        color_r += r
        color_g += g
        color_b += b
        rng_state = cuda_xorshift(rng_state)

    color_r /= samples_per_pixel
    color_g /= samples_per_pixel
    color_b /= samples_per_pixel

    color_r = cuda_tonemap(color_r)
    color_g = cuda_tonemap(color_g)
    color_b = cuda_tonemap(color_b)

    output[pixel_idx * 3 + 0] = min(255, max(0, int(color_r * 255)))
    output[pixel_idx * 3 + 1] = min(255, max(0, int(color_g * 255)))
    output[pixel_idx * 3 + 2] = min(255, max(0, int(color_b * 255)))


@cuda.jit(device=True)
def cuda_xorshift(state):
    state ^= state << 13
    state ^= state >> 17
    state ^= state << 5
    return state & 0xffffffff


@cuda.jit(device=True)
def cuda_random(state):
    return (state & 0xffffff) / 16777216.0


@cuda.jit(device=True)
def cuda_tonemap(x):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return (x * (a * x + b)) / (x * (c * x + d) + e)


@cuda.jit(device=True)
def cuda_get_ray(camera_data, u, v):
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

    dir_x = llc_x + u * h_x + v * v_x - origin_x
    dir_y = llc_y + u * h_y + v * v_y - origin_y
    dir_z = llc_z + u * h_z + v * v_z - origin_z

    length = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
    if length > 0:
        dir_x /= length
        dir_y /= length
        dir_z /= length

    return (origin_x, origin_y, origin_z), (dir_x, dir_y, dir_z)


@cuda.jit(device=True)
def cuda_refract_path(incident_x, incident_y, incident_z, normal_x, normal_y, normal_z, ni_over_nt):
    cos_i = -(incident_x * normal_x + incident_y * normal_y + incident_z * normal_z)
    sin2_t = ni_over_nt * ni_over_nt * (1.0 - cos_i * cos_i)

    if sin2_t > 1.0:
        return False, 0.0, 0.0, 0.0

    cos_t = math.sqrt(1.0 - sin2_t)
    factor1 = ni_over_nt
    factor2 = ni_over_nt * cos_i - cos_t

    refracted_x = factor1 * incident_x + factor2 * normal_x
    refracted_y = factor1 * incident_y + factor2 * normal_y
    refracted_z = factor1 * incident_z + factor2 * normal_z

    return True, refracted_x, refracted_y, refracted_z


@cuda.jit(device=True)
def cuda_fresnel_schlick(cos_theta, f0):
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0)


@cuda.jit(device=True)
def cuda_sample_hemisphere_cosine(normal_x, normal_y, normal_z, rng_state):
    r1 = cuda_random(rng_state)
    rng_state = cuda_xorshift(rng_state)
    r2 = cuda_random(rng_state)
    rng_state = cuda_xorshift(rng_state)

    cos_theta = math.sqrt(r1)
    sin_theta = math.sqrt(1.0 - r1)
    phi = 2.0 * math.pi * r2

    x = sin_theta * math.cos(phi)
    y = sin_theta * math.sin(phi)
    z = cos_theta

    if abs(normal_z) > 0.9:
        nt_x = 1.0
        nt_y = 0.0
        nt_z = 0.0
    else:
        nt_x = 0.0
        nt_y = 0.0
        nt_z = 1.0

    u_x = nt_y * normal_z - nt_z * normal_y
    u_y = nt_z * normal_x - nt_x * normal_z
    u_z = nt_x * normal_y - nt_y * normal_x

    u_len = math.sqrt(u_x * u_x + u_y * u_y + u_z * u_z)
    u_x /= u_len
    u_y /= u_len
    u_z /= u_len

    v_x = normal_y * u_z - normal_z * u_y
    v_y = normal_z * u_x - normal_x * u_z
    v_z = normal_x * u_y - normal_y * u_x

    dir_x = x * u_x + y * v_x + z * normal_x
    dir_y = x * u_y + y * v_y + z * normal_y
    dir_z = x * u_z + y * v_z + z * normal_z

    return (dir_x, dir_y, dir_z), rng_state


@cuda.jit(device=True)
def cuda_sample_light_direction(light_data, hit_point, rng_state):
    num_lights = int(light_data[0])
    if num_lights == 0:
        return (0.0, 0.0, 0.0), 0.0, rng_state

    light_idx = int(cuda_random(rng_state) * num_lights)
    if light_idx >= num_lights:
        light_idx = num_lights - 1
    rng_state = cuda_xorshift(rng_state)

    light_offset = 1 + light_idx * 3
    light_x = light_data[light_offset + 0]
    light_y = light_data[light_offset + 1]
    light_z = light_data[light_offset + 2]

    dx = light_x - hit_point[0]
    dy = light_y - hit_point[1]
    dz = light_z - hit_point[2]

    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance > 0.001:
        dx /= distance
        dy /= distance
        dz /= distance

    pdf = 1.0 / num_lights
    return (dx, dy, dz), pdf, rng_state


# cuda_path_tracer.py의 cuda_trace_path 함수에서 재질별 샘플링 부분 완전 수정

@cuda.jit(device=True)
def cuda_trace_path(scene_data, light_data, texture_data, texture_info,
                    ray_origin, ray_direction, max_depth, rng_state):
    color_r = 0.0
    color_g = 0.0
    color_b = 0.0

    throughput_r = 1.0
    throughput_g = 1.0
    throughput_b = 1.0

    current_origin = ray_origin
    current_direction = ray_direction

    for depth in range(max_depth):
        hit, hit_t, hit_point, hit_normal, material_data, uv_coords = cuda_scene_hit(
            scene_data, current_origin, current_direction, 0.001, 1000000.0
        )

        if not hit:
            sky_color = 0.1
            color_r += throughput_r * sky_color
            color_g += throughput_g * sky_color
            color_b += throughput_b * sky_color
            break

        mat_color_r = material_data[0]
        mat_color_g = material_data[1]
        mat_color_b = material_data[2]
        mat_diffuse = material_data[3]
        mat_specular = material_data[4]
        mat_reflective = material_data[5]
        mat_refractive = material_data[6]
        mat_ior = material_data[7]
        has_texture = material_data[8] > 0.5
        texture_id = int(material_data[9])

        if has_texture and texture_id >= 0 and texture_id < len(texture_info) // 3:
            tex_start = int(texture_info[texture_id * 3 + 0])
            tex_width = int(texture_info[texture_id * 3 + 1])
            tex_height = int(texture_info[texture_id * 3 + 2])

            tex_r, tex_g, tex_b = cuda_sample_texture(
                texture_data, tex_start, tex_width, tex_height, uv_coords[0], uv_coords[1]
            )
            mat_color_r = tex_r
            mat_color_g = tex_g
            mat_color_b = tex_b

        # ===== 모든 재질에 직접 조명 적용 =====
        if len(light_data) > 1:
            light_dir, light_pdf, rng_state = cuda_sample_light_direction(light_data, hit_point, rng_state)

            if light_pdf > 0.0:
                shadow_origin = (
                    hit_point[0] + hit_normal[0] * 0.001,
                    hit_point[1] + hit_normal[1] * 0.001,
                    hit_point[2] + hit_normal[2] * 0.001
                )

                shadow_hit, shadow_t, _, _, _, _ = cuda_scene_hit(
                    scene_data, shadow_origin, light_dir, 0.001, 1000000.0
                )

                if not shadow_hit:
                    cos_theta = max(0.0, light_dir[0] * hit_normal[0] +
                                    light_dir[1] * hit_normal[1] +
                                    light_dir[2] * hit_normal[2])

                    # 재질별 조명 강도 조정
                    if mat_refractive > 0.5:
                        # 투명 재질: 더 강한 직접 조명
                        light_intensity = 4.0
                        light_multiplier = 0.6
                    elif mat_reflective > 0.7:
                        # 반사 재질
                        light_intensity = 2.5
                        light_multiplier = 0.8
                    else:
                        # 일반 재질
                        light_intensity = 2.0
                        light_multiplier = 1.0

                    contrib_r = mat_color_r * mat_diffuse * cos_theta * light_intensity * light_multiplier / light_pdf
                    contrib_g = mat_color_g * mat_diffuse * cos_theta * light_intensity * light_multiplier / light_pdf
                    contrib_b = mat_color_b * mat_diffuse * cos_theta * light_intensity * light_multiplier / light_pdf

                    color_r += throughput_r * contrib_r
                    color_g += throughput_g * contrib_g
                    color_b += throughput_b * contrib_b

        # Russian Roulette
        if depth >= 3:
            survival_prob = max(0.1, 0.299 * throughput_r + 0.587 * throughput_g + 0.114 * throughput_b)
            if cuda_random(rng_state) > survival_prob:
                break
            rng_state = cuda_xorshift(rng_state)
            throughput_r /= survival_prob
            throughput_g /= survival_prob
            throughput_b /= survival_prob

        # ===== 핵심 수정: 재질별 샘플링 전략 =====
        sample_choice = cuda_random(rng_state)
        rng_state = cuda_xorshift(rng_state)

        if mat_refractive > 0.1:
            # ===== 투명 재질: 다중 이벤트 샘플링 =====

            # 확률 분배: 굴절 60%, 반사 25%, 확산 15%
            refraction_prob = 0.6
            reflection_prob = 0.25
            diffuse_prob = 0.15

            if sample_choice < refraction_prob:
                # === 굴절 이벤트 ===
                cos_i = max(0.0, -(current_direction[0] * hit_normal[0] +
                                   current_direction[1] * hit_normal[1] +
                                   current_direction[2] * hit_normal[2]))

                entering = cos_i > 0.0

                if entering:
                    eta = 1.0 / mat_ior
                    outward_normal = hit_normal
                else:
                    eta = mat_ior
                    outward_normal = (-hit_normal[0], -hit_normal[1], -hit_normal[2])
                    cos_i = -cos_i

                refracted, refract_x, refract_y, refract_z = cuda_refract_path(
                    current_direction[0], current_direction[1], current_direction[2],
                    outward_normal[0], outward_normal[1], outward_normal[2], eta
                )

                if refracted:
                    if entering:
                        current_origin = (
                            hit_point[0] - hit_normal[0] * 0.001,
                            hit_point[1] - hit_normal[1] * 0.001,
                            hit_point[2] - hit_normal[2] * 0.001
                        )
                    else:
                        current_origin = (
                            hit_point[0] + hit_normal[0] * 0.001,
                            hit_point[1] + hit_normal[1] * 0.001,
                            hit_point[2] + hit_normal[2] * 0.001
                        )

                    current_direction = (refract_x, refract_y, refract_z)

                    # PDF 보정
                    throughput_r *= mat_refractive / refraction_prob
                    throughput_g *= mat_refractive / refraction_prob
                    throughput_b *= mat_refractive / refraction_prob
                else:
                    # 전반사 -> 반사로 처리
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

                    throughput_r *= 0.9
                    throughput_g *= 0.9
                    throughput_b *= 0.9

            elif sample_choice < refraction_prob + reflection_prob:
                # === 반사 이벤트 ===
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

                # PDF 보정
                throughput_r *= mat_color_r * 0.9 / reflection_prob
                throughput_g *= mat_color_g * 0.9 / reflection_prob
                throughput_b *= mat_color_b * 0.9 / reflection_prob

            else:
                # === 확산 반사 이벤트 (핵심 추가!) ===
                next_dir, rng_state = cuda_sample_hemisphere_cosine(
                    hit_normal[0], hit_normal[1], hit_normal[2], rng_state
                )

                current_origin = (
                    hit_point[0] + hit_normal[0] * 0.001,
                    hit_point[1] + hit_normal[1] * 0.001,
                    hit_point[2] + hit_normal[2] * 0.001
                )
                current_direction = next_dir

                # PDF 보정 - 이 부분이 빛알갱이를 만듦!
                throughput_r *= mat_color_r * mat_diffuse * 3.0 / diffuse_prob
                throughput_g *= mat_color_g * mat_diffuse * 3.0 / diffuse_prob
                throughput_b *= mat_color_b * mat_diffuse * 3.0 / diffuse_prob

        elif mat_reflective > 0.5:
            # ===== 반사 재질 =====
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

            throughput_r *= mat_color_r * mat_reflective
            throughput_g *= mat_color_g * mat_reflective
            throughput_b *= mat_color_b * mat_reflective

        else:
            # ===== 확산 반사 재질 =====
            next_dir, rng_state = cuda_sample_hemisphere_cosine(
                hit_normal[0], hit_normal[1], hit_normal[2], rng_state
            )

            current_origin = (
                hit_point[0] + hit_normal[0] * 0.001,
                hit_point[1] + hit_normal[1] * 0.001,
                hit_point[2] + hit_normal[2] * 0.001
            )
            current_direction = next_dir

            throughput_r *= mat_color_r * mat_diffuse
            throughput_g *= mat_color_g * mat_diffuse
            throughput_b *= mat_color_b * mat_diffuse

        if max(throughput_r, throughput_g, throughput_b) < 0.001:
            break

    return color_r, color_g, color_b

@cuda.jit(device=True)
def cuda_sample_texture(texture_data, texture_start, texture_width, texture_height, u, v):
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))

    iu = int(u * (texture_width - 1))
    iv = int((1.0 - v) * (texture_height - 1))

    iu = max(0, min(texture_width - 1, iu))
    iv = max(0, min(texture_height - 1, iv))

    pixel_idx = iv * texture_width + iu
    base_idx = texture_start + pixel_idx * 3

    if base_idx + 2 < len(texture_data):
        r = texture_data[base_idx + 0] / 255.0
        g = texture_data[base_idx + 1] / 255.0
        b = texture_data[base_idx + 2] / 255.0
        return r, g, b
    else:
        return 1.0, 1.0, 1.0


@cuda.jit(device=True)
def cuda_scene_hit(scene_data, ray_origin, ray_direction, t_min, t_max):
    closest_t = t_max
    hit = False
    hit_point = (0.0, 0.0, 0.0)
    hit_normal = (0.0, 1.0, 0.0)
    material_data = (0.5, 0.5, 0.5, 0.8, 0.2, 0.0, 0.0, 1.0, 0.0, -1.0)
    uv_coords = (0.0, 0.0)

    offset = 0

    # Planes 처리
    num_planes = int(scene_data[offset])
    offset += 1

    for i in range(num_planes):
        plane_offset = offset + i * 20
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

    # Spheres 처리
    num_spheres = int(scene_data[offset])
    offset += 1

    for i in range(num_spheres):
        sphere_offset = offset + i * 12
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
        mat_refractive = scene_data[sphere_offset + 10]
        mat_ior = scene_data[sphere_offset + 11]

        oc_x = ray_origin[0] - center_x
        oc_y = ray_origin[1] - center_y
        oc_z = ray_origin[2] - center_z

        a = (ray_direction[0] * ray_direction[0] + ray_direction[1] * ray_direction[1] + ray_direction[2] *
             ray_direction[2])
        b = (oc_x * ray_direction[0] + oc_y * ray_direction[1] + oc_z * ray_direction[2])
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
                uv_coords = (0.0, 0.0)
                material_data = (color_r, color_g, color_b, mat_diffuse, mat_specular,
                                 mat_reflective, mat_refractive, mat_ior, 0.0, -1.0)

    offset += num_spheres * 12

    # Triangles 처리
    num_triangles = int(scene_data[offset])
    offset += 1

    for i in range(num_triangles):
        tri_offset = offset + i * 26
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

        h_x = ray_direction[1] * edge2_z - ray_direction[2] * edge2_y
        h_y = ray_direction[2] * edge2_x - ray_direction[0] * edge2_z
        h_z = ray_direction[0] * edge2_y - ray_direction[1] * edge2_x

        a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z

        if abs(a) < 1e-6:
            continue

        f = 1.0 / a
        s_x = ray_origin[0] - v0_x
        s_y = ray_origin[1] - v0_y
        s_z = ray_origin[2] - v0_z

        u = f * (s_x * h_x + s_y * h_y + s_z * h_z)
        if u < 0.0 or u > 1.0:
            continue

        q_x = s_y * edge1_z - s_z * edge1_y
        q_y = s_z * edge1_x - s_x * edge1_z
        q_z = s_x * edge1_y - s_y * edge1_x

        v = f * (ray_direction[0] * q_x + ray_direction[1] * q_y + ray_direction[2] * q_z)
        if v < 0.0 or u + v > 1.0:
            continue

        t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z)

        if t_min < t < closest_t:
            closest_t = t
            hit = True

            hit_x = ray_origin[0] + t * ray_direction[0]
            hit_y = ray_origin[1] + t * ray_direction[1]
            hit_z = ray_origin[2] + t * ray_direction[2]

            dot_product = normal_x * ray_direction[0] + normal_y * ray_direction[1] + normal_z * ray_direction[2]
            if dot_product > 0:
                normal_x = -normal_x
                normal_y = -normal_y
                normal_z = -normal_z

            hit_point = (hit_x, hit_y, hit_z)
            hit_normal = (normal_x, normal_y, normal_z)

            w = 1.0 - u - v
            final_u = w * uv0_u + u * uv1_u + v * uv2_u
            final_v = w * uv0_v + u * uv1_v + v * uv2_v
            uv_coords = (final_u, final_v)

            material_data = (color_r, color_g, color_b, mat_diffuse, mat_specular,
                             mat_reflective, 0.0, 1.0, has_texture, texture_id)

    return hit, closest_t, hit_point, hit_normal, material_data, uv_coords


class CUDAPathTracer(BaseRenderer):
    """CUDA 가속 Path Tracer - Global Illumination 지원"""

    def __init__(self):
        super().__init__("cuda_path_raytracer")
        self._check_cuda_available()
        self.frame_count = 0

    def _check_cuda_available(self):
        try:
            cuda.select_device(0)
            print(f"CUDA 디바이스 감지: {cuda.get_current_device().name.decode()}")
        except Exception as e:
            raise RuntimeError(f"CUDA를 사용할 수 없습니다: {e}")

    def get_capabilities(self) -> List[str]:
        return [
            "path_tracing",
            "global_illumination",
            "monte_carlo_integration",
            "color_bleeding",
            "physically_based_rendering",
            "shadows",
            "reflection",
            "refraction",
            "fresnel_effects",
            "specular_highlights",
            "textures",
            "gpu_acceleration",
            "anti_aliasing",
            "hdr_rendering",
            "tone_mapping",
            "russian_roulette",
            "importance_sampling"
        ]

    def render(self, scene: Scene, camera: Camera, settings: RenderSettings) -> Image.Image:
        start_time = time.time()

        print(f"CUDA Path Tracing 시작: {settings.width}x{settings.height}")
        print(f"샘플 수: {settings.samples_per_pixel}")
        print(f"최대 깊이: {settings.max_depth}")

        scene_data = self._prepare_scene_data(scene)
        camera_data = self._prepare_camera_data(camera)
        light_data = self._prepare_light_data(scene)
        texture_data, texture_info = self._prepare_texture_data(scene)

        print(f"씬 객체: Planes={len([o for o in scene.objects if isinstance(o, Plane)])}, "
              f"Spheres={len([o for o in scene.objects if isinstance(o, Sphere)])}, "
              f"Triangles={len([o for o in scene.objects if isinstance(o, Triangle)])}")

        output_size = settings.width * settings.height * 3
        d_output = cuda.device_array(output_size, dtype=np.uint8)
        d_scene_data = cuda.to_device(scene_data)
        d_camera_data = cuda.to_device(camera_data)
        d_light_data = cuda.to_device(light_data)
        d_texture_data = cuda.to_device(texture_data)
        d_texture_info = cuda.to_device(texture_info)

        threads_per_block = (16, 16)
        blocks_per_grid_x = (settings.width + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (settings.height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        print(f"CUDA 설정: {blocks_per_grid} blocks, {threads_per_block} threads/block")

        cuda_path_trace_kernel[blocks_per_grid, threads_per_block](
            d_output, d_scene_data, d_camera_data, d_light_data, d_texture_data, d_texture_info,
            settings.width, settings.height, settings.samples_per_pixel, settings.max_depth, self.frame_count
        )

        output_array = d_output.copy_to_host()
        image_array = output_array.reshape((settings.height, settings.width, 3))
        image_array = np.flip(image_array, axis=0)

        self.frame_count += 1

        end_time = time.time()
        elapsed = end_time - start_time
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"CUDA Path Tracing 완료: {minutes}분 {seconds:.2f}초")

        return Image.fromarray(image_array, 'RGB')

    def _prepare_scene_data(self, scene: Scene) -> np.ndarray:
        data = []
        texture_map = {}
        texture_paths = []

        for obj in scene.objects:
            if hasattr(obj, 'material') and obj.material.texture is not None:
                texture_path = getattr(obj.material.texture, 'path', None)
                if texture_path and texture_path not in texture_paths:
                    texture_paths.append(texture_path)

        texture_paths.sort()
        for i, texture_path in enumerate(texture_paths):
            texture_map[texture_path] = i

        planes = []
        spheres = []
        triangles = []

        for obj in scene.objects:
            if isinstance(obj, Plane):
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
                mat = obj.material
                spheres.extend([
                    obj.center.x, obj.center.y, obj.center.z, obj.radius,
                    mat.color.x, mat.color.y, mat.color.z,
                    mat.diffuse, mat.specular, mat.reflective,
                    getattr(mat, 'refractive', 0.0), getattr(mat, 'ior', 1.0)
                ])

            elif isinstance(obj, Triangle):
                mat = obj.material
                has_texture = 1.0 if mat.texture is not None else 0.0
                tex_id = -1.0
                if mat.texture is not None:
                    texture_path = getattr(mat.texture, 'path', None)
                    if texture_path in texture_map:
                        tex_id = float(texture_map[texture_path])

                uv0_u = obj.uv0[0] if obj.uv0 is not None else 0.0
                uv0_v = obj.uv0[1] if obj.uv0 is not None else 0.0
                uv1_u = obj.uv1[0] if obj.uv1 is not None else 1.0
                uv1_v = obj.uv1[1] if obj.uv1 is not None else 0.0
                uv2_u = obj.uv2[0] if obj.uv2 is not None else 1.0
                uv2_v = obj.uv2[1] if obj.uv2 is not None else 1.0

                triangles.extend([
                    obj.v0.x, obj.v0.y, obj.v0.z,
                    obj.v1.x, obj.v1.y, obj.v1.z,
                    obj.v2.x, obj.v2.y, obj.v2.z,
                    obj.normal.x, obj.normal.y, obj.normal.z,
                    mat.color.x, mat.color.y, mat.color.z,
                    mat.diffuse, mat.specular, mat.reflective,
                    has_texture, tex_id,
                    uv0_u, uv0_v, uv1_u, uv1_v, uv2_u, uv2_v
                ])

        data.append(len(planes) // 20)
        data.extend(planes)
        data.append(len(spheres) // 12)
        data.extend(spheres)
        data.append(len(triangles) // 26)
        data.extend(triangles)

        print(f"=== GPU 씬 데이터 (Path Tracer) ===")
        print(f"Planes: {len(planes) // 20}개")
        print(f"Spheres: {len(spheres) // 12}개")
        print(f"Triangles: {len(triangles) // 26}개")

        return np.array(data, dtype=np.float32)

    def _prepare_texture_data(self, scene: Scene):
        texture_data = []
        texture_info = []
        current_offset = 0

        texture_paths = []
        for obj in scene.objects:
            if hasattr(obj, 'material') and obj.material.texture is not None:
                texture_path = getattr(obj.material.texture, 'path', None)
                if texture_path and texture_path not in texture_paths:
                    texture_paths.append(texture_path)

        texture_paths.sort()

        for i, texture_path in enumerate(texture_paths):
            try:
                img = Image.open(texture_path).convert("RGB")
                width, height = img.size
                pixels = np.array(img)
                pixel_data = pixels.reshape(-1).astype(np.uint8)

                texture_info.extend([current_offset, width, height])
                texture_data.extend(pixel_data.tolist())
                current_offset += len(pixel_data)

            except Exception as e:
                print(f"텍스처 로드 실패 {texture_path}: {e}")
                texture_info.extend([current_offset, 1, 1])
                texture_data.extend([255, 255, 255])
                current_offset += 3

        return np.array(texture_data, dtype=np.uint8), np.array(texture_info, dtype=np.int32)

    def _prepare_camera_data(self, camera: Camera) -> np.ndarray:
        return np.array([
            camera.origin.x, camera.origin.y, camera.origin.z,
            camera.lower_left_corner.x, camera.lower_left_corner.y, camera.lower_left_corner.z,
            camera.horizontal.x, camera.horizontal.y, camera.horizontal.z,
            camera.vertical.x, camera.vertical.y, camera.vertical.z
        ], dtype=np.float32)

    def _prepare_light_data(self, scene: Scene) -> np.ndarray:
        data = [len(scene.lights)]
        for light_pos in scene.lights:
            data.extend([light_pos.x, light_pos.y, light_pos.z])
        return np.array(data, dtype=np.float32)


# 렌더러 등록
RendererFactory.register("cuda_path_raytracer", CUDAPathTracer)