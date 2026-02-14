from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

class AIModel(ABC):
    @abstractmethod
    def generate_response(self, messages, system_prompt, file_data=None):        pass
    
    @abstractmethod
    def generate_image(self, prompt: str) -> bytes:
        pass
    
    @abstractmethod
    def generate_video(self, prompt: str, image_data: bytes = None) -> bytes:
        pass