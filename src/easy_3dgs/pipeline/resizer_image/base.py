from abc import ABC, abstractmethod

class BaseResizer(ABC):
    @abstractmethod
    def main(self, sfm_dir: str, magnifications: list[int]):
        pass
