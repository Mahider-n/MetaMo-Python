from abc import ABC, abstractmethod
from typing import List, Optional

from core.state import DocumentChunk, Paper


class TextExtractor(ABC):
    @abstractmethod
    def extract(self, path: str) -> str:
        ...


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str, source_path: str) -> List[DocumentChunk]:
        ...


class PaperRepository(ABC):
    @abstractmethod
    def save(self, paper: Paper) -> str:
        ...

    @abstractmethod
    def load(self, paper_id: str) -> Optional[Paper]:
        ...

    @abstractmethod
    def list_papers(self) -> List[Paper]:
        ...

    @abstractmethod
    def delete(self, paper_id: str) -> bool:
        ...
