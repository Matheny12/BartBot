from ai_models import AIModel
from typing import List, Dict, Optional, Generator
from gpt4all import GPT4All
from diffusers import StableDiffusionPipeline
import io
import os
import streamlit as st

class BartBotModel(AIModel):
    @st.cache_resource
    def _get_llm(_self):
        model_name = "Llama-3.2-1B-Instruct-Q4_0.gguf"
        return GPT4All(model_name=model_name, allow_download=True)
    
    def __init__(self):
        self.llm = self._get_llm()
        self.image_model = None

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
        import os
        from huggingface_hub import InferenceClient
        import io
        import time

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            raise Exception("HF_TOKEN not found. Please add your Hugging Face token to Streamlit secrets or environment variables for image generation.")

        models_to_try = [
            ("stabilityai/stable-diffusion-2-1", {"num_inference_steps": 20}),
            ("black-forest-labs/FLUX.1-schnell", {"num_inference_steps": 4}),
            ("ByteDance/SDXL-Lightning", {"num_inference_steps": 4}),
        ]


        client = InferenceClient(api_key=hf_token)
        last_error = None

        for model_name, params in models_to_try:
            try:
                image = client.text_to_image(
                    prompt,
                    model=model_name,
                    guidance_scale=7.5,
                    width=512,
                    height=512,
                    **params
                )
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                return img_byte_arr.getvalue()
            except Exception as e:
                last_error = str(e)
                if "timeout" in str(e).lower() or "cold state" in str(e).lower():
                    continue
                continue
        raise Exception(f"Failed to generate image after trying multiple models. Last error: {last_error}")



    def generate_video(self, prompt: str, image_data: bytes = None) -> bytes:
        import os
        import requests
        import base64
        import time
        
        if not image_data:
            raise NotImplementedError(
                "Text-to-video is not supported. Please upload an image first, then use /video to animate it."
            )
        
        modelslab_key = os.getenv("MODELSLAB_API_KEY") or st.secrets.get("MODELSLAB_API_KEY")
        if not modelslab_key:
            raise Exception(
                "MODELSLAB_API_KEY not found. Please add your ModelsLab API key to Streamlit secrets. "
                "Get your free API key at https://modelslab.com/developers"
            )
        
        try:
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            
            payload = {
                "key": modelslab_key,
                "model_id": "svd",
                "init_image": f"data:image/png;base64,{encoded_image}",
                "height": 512,
                "width": 512,
                "num_frames": 25,
                "num_inference_steps": 20,
                "min_guidance_scale": 1,
                "max_guidance_scale": 3,
                "motion_bucket_id": 20,
                "noise_aug_strength": 0.02,
                "strength": 0.7,
                "base64": False,
                "webhook": None,
                "track_id": None
            }
            
            response = requests.post(
                "https://modelslab.com/api/v1/enterprise/video/img2video",
                json=payload,
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"ModelsLab API error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            if result.get("status") == "processing":
                fetch_url = result.get("fetch_result")
                if fetch_url:
                    for attempt in range(30):
                        time.sleep(3)
                        fetch_response = requests.get(fetch_url)
                        fetch_data = fetch_response.json()
                        
                        if fetch_data.get("status") == "success":
                            video_url = fetch_data.get("output", [None])[0]
                            if video_url:
                                video_response = requests.get(video_url)
                                return video_response.content
                        elif fetch_data.get("status") == "failed":
                            raise Exception(f"Video generation failed: {fetch_data.get('message', 'Unknown error')}")
            
            elif result.get("status") == "success":
                video_url = result.get("output", [None])[0]
                if video_url:
                    video_response = requests.get(video_url)
                    return video_response.content
            
            raise Exception(f"Unexpected response: {result}")
            
        except Exception as e:
            raise Exception(f"Failed to generate video: {str(e)}")
