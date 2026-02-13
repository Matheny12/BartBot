from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class AIModel(ABC):
    @abstractmethod
    def generate_response(self, messages: List[Dict], system_prompt: str, file_data: Optional[Dict] = None) -> str:
        pass
    
    @abstractmethod
    def generate_image(self, prompt: str) -> bytes:
        pass