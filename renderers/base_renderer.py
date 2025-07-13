from abc import ABC, abstractmethod
from typing import List
from PIL import Image
from core.scene import Scene, RenderSettings


class BaseRenderer(ABC):
    """모든 렌더러가 구현해야 하는 베이스 클래스"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def render(self, scene: Scene, camera, settings: RenderSettings) -> Image.Image:
        """장면을 렌더링하여 PIL Image를 반환"""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """이 렌더러가 지원하는 기능들을 반환"""
        pass

    def get_name(self) -> str:
        return self.name

    def supports(self, feature: str) -> bool:
        """특정 기능을 지원하는지 확인"""
        return feature in self.get_capabilities()


class RendererFactory:
    """렌더러 팩토리 클래스"""

    _renderers = {}

    @classmethod
    def register(cls, name: str, renderer_class):
        """새로운 렌더러를 등록"""
        cls._renderers[name] = renderer_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseRenderer:
        """등록된 렌더러를 생성"""
        if name not in cls._renderers:
            raise ValueError(f"Unknown renderer: {name}")
        return cls._renderers[name](**kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """사용 가능한 렌더러 목록 반환"""
        return list(cls._renderers.keys())