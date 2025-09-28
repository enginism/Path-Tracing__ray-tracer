import time
import argparse
from core.scene import RenderSettings
from scene_builders.custom_scene_builder import CustomSceneBuilder
from renderers.base_renderer import RendererFactory

# 렌더러 모듈들 import (등록을 위해)
import renderers.cpu_renderer

# CUDA 렌더러 import (CUDA 사용 가능한 경우에만)
try:
    import renderers.cuda_renderer
    import renderers.cuda_texture_renderer  # 텍스처 지원 CUDA 렌더러
    import renderers.cuda_path_tracer  # Path Tracer 추가

    CUDA_AVAILABLE = True
    print("CUDA 렌더러 사용 가능 (텍스처 + Path Tracing 지원 포함)")
except Exception as e:
    CUDA_AVAILABLE = False
    print(f"CUDA 렌더러 사용 불가: {e}")


def main():
    parser = argparse.ArgumentParser(description='Modular Ray Tracer with Path Tracing')
    parser.add_argument('--renderer', '-r',
                        choices=RendererFactory.list_available(),
                        default='cuda_texture_raytracer',  # 올바른 이름으로 수정
                        help='렌더러 선택')
    parser.add_argument('--scene',
                        choices=['original', 'custom'],
                        default='custom',
                        help='씬 선택: custom (내 코넬박스)')
    parser.add_argument('--width', '-w', type=int, default=2000,
                        help='이미지 가로 크기')
    parser.add_argument('--height', type=int, default=1500,
                        help='이미지 세로 크기')
    parser.add_argument('--samples', '-s', type=int, default=25,
                        help='픽셀당 샘플 수')
    parser.add_argument('--depth', '-d', type=int, default=16,
                        help='최대 재귀 깊이')
    parser.add_argument('--output', '-o', default='output.png',
                        help='출력 파일명')
    parser.add_argument('--path-samples', type=int, default=1024,
                        help='Path Tracer 전용 샘플 수 (기본값: 64)')

    args = parser.parse_args()

    # 샘플 수 계산 로직 단순화
    if args.renderer == 'cuda_path_raytracer':
        effective_samples = args.path_samples  # 직접 사용
        print(f"Path Tracer 모드: {effective_samples} 샘플")
    else:
        effective_samples = args.samples
        print(f"Ray Tracer 모드: {effective_samples} 샘플")

    # 렌더링 설정
    settings = RenderSettings(
        width=args.width,
        height=args.height,
        samples_per_pixel=effective_samples,
        max_depth=args.depth
    )

    # 씬 생성
    print(f"장면 생성 중: {args.scene}")
    scene_builder = CustomSceneBuilder()
    scene = scene_builder.build_scene()

    # 카메라는 화면 비율에 맞춰 생성
    aspect_ratio = args.width / args.height
    camera = scene_builder.create_camera(aspect_ratio)

    # 렌더러 생성
    print(f"렌더러 생성: {args.renderer}")
    renderer = RendererFactory.create(args.renderer)

    print(f"지원 기능: {', '.join(renderer.get_capabilities())}")

    # 렌더링 방식에 따른 예상 시간 안내
    if args.renderer == 'cuda_path_raytracer':  
        estimated_time = (effective_samples / 64) * 15  # 64 샘플 기준 약 15초
        print(f"예상 렌더링 시간: 약 {estimated_time:.0f}초 (Global Illumination)")
    elif 'cuda' in args.renderer:
        print("예상 렌더링 시간: 3-10초 (GPU 가속)")
    else:
        print("예상 렌더링 시간: 30-60초 (CPU)")

    # 렌더링 실행
    start_time = time.time()
    image = renderer.render(scene, camera, settings)
    end_time = time.time()

    # 결과 저장
    image.save(args.output)
    print(f"이미지 저장: {args.output}")

    # 실행 시간 출력
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"총 실행 시간: {minutes}분 {seconds:.2f}초")

    # 성능 분석
    if args.renderer == 'cuda_path_raytracer':  
        rays_per_pixel = effective_samples * args.depth
        total_rays = args.width * args.height * rays_per_pixel
        rays_per_second = total_rays / elapsed
        print(f"성능: {rays_per_second / 1e6:.2f}M rays/sec ({total_rays / 1e6:.1f}M rays total)")

    # 품질 정보
    if 'path_tracer' in args.renderer:
        print("렌더링 품질: Global Illumination (최고 품질)")
    elif 'cuda_texture' in args.renderer:
        print("렌더링 품질: Whitted Ray Tracing + 텍스처 (고품질)")
    elif 'cuda' in args.renderer:
        print("렌더링 품질: GPU 가속 Ray Tracing (중품질)")
    else:
        print("렌더링 품질: CPU Ray Tracing (기본품질)")

    # 이미지 표시 (선택사항)
    try:
        image.show()
    except:
        print("이미지 표시 불가")


if __name__ == "__main__":
    main()

