from ai_models import AIModel
from typing import List, Dict, Optional, Generator
from gpt4all import GPT4All
import io
import os
import streamlit as st
import requests
import time
import imghdr
from PIL import Image as PILImage

class BartBotModel(AIModel):
    @st.cache_resource
    def _get_llm(_self):
        model_name = "Llama-3.2-1B-Instruct-Q4_0.gguf"
        return GPT4All(model_name=model_name, allow_download=True)
    
    def __init__(self):
        self.llm = self._get_llm()
        self.api_key = st.secrets.get("GEMINI_KEY") or os.getenv("GEMINI_KEY")
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def generate_response(self, messages: List[Dict], system_prompt: str, file_data: Optional[Dict] = None) -> Generator:        
        with self.llm.chat_session(system_prompt):
            user_input = messages[-1]["content"]   
            response_generator = self.llm.generate(
                user_input, 
                max_tokens=1024, 
                streaming=True
            )
            for token in response_generator:
                yield token

    def generate_image(self, prompt: str) -> bytes:
        from huggingface_hub import InferenceClient
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            raise Exception("HF_TOKEN missing for image generation.")
        
        client = InferenceClient(api_key=hf_token)
        image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-schnell")
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    
    def generate_video(self, prompt: str, image_data: bytes = None) -> bytes:
        from google.genai import types
        if not image_data:
            raise ValueError("Please upload an image first.")

        try:
            img_format = imghdr.what(None, h=image_data)
            if img_format == 'jpeg':
                mime_type = "image/jpeg"
            elif img_format == 'png':
                mime_type = "image/png"
            elif img_format == 'gif':
                mime_type = "image/gif"
            elif img_format == 'webp':
                mime_type = "image/webp"
            else:
                try:
                    img = PILImage.open(io.BytesIO(image_data))
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    image_data = buffered.getvalue()
                    mime_type = "image/png"
                except:
                    mime_type = "image/png"
            
            operation = self.client.models.generate_videos(
                model="veo-3.1-fast-generate-preview",
                prompt=prompt or "animate this",
                image=types.Image(image_bytes=image_data, mime_type=mime_type),
                config=types.GenerateVideosConfig(resolution="720p")
            )

            while not operation.done:
                time.sleep(5)
                operation = self.client.operations.get(operation)
            
            if operation.result and operation.result.generated_videos:
                uri = operation.result.generated_videos[0].video.uri
                return requests.get(uri).content
            
            raise Exception("No video was generated.")
        except Exception as e:
            raise Exception(f"Video generation failed: {e}")