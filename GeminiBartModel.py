from ai_models import AIModel
from google import genai
from google.genai import types
from typing import List, Dict, Optional

class GeminiModel(AIModel):
    def __init__(self, api_key: str, bot_name: str = "Bartholemew"):
        self.client = genai.Client(api_key=api_key)
        self.bot_name = bot_name
    
    def generate_response(self, messages: List[Dict], system_prompt: str, file_data: Optional[Dict] = None):
        formatted_history = []
        for m in messages[:-1]:
            gemini_role = "model" if m["role"] == "assistant" else "user"
            clean_text = m["content"] if not str(m["content"]).startswith("IMAGE_DATA:") else "[Image]"
            formatted_history.append({"role": gemini_role, "parts": [{"text": clean_text}]})
        
        chat_session = self.client.chats.create(
            model="gemini-2.5-flash-lite",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
            history=formatted_history
        )

        last_prompt = messages[-1]["content"]
        content_to_send = [last_prompt]
        
        if file_data:
            content_to_send.append(
                types.Part.from_bytes(
                    data=file_data["bytes"],
                    mime_type=file_data["mime"]
                )
            )
            
        response_stream = chat_session.send_message_stream(content_to_send)
        
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    def generate_image(self, prompt: str) -> bytes:
        safe_prompt = self._refine_prompt(prompt)
        
        model_options = ['imagen-4.0-generate-001']
        last_error = ""
        
        for model_id in model_options:
            try:
                response = self.client.models.generate_images(
                    model=model_id,
                    prompt=safe_prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="1:1",
                        person_generation="ALLOW_ADULT",
                        safety_filter_level="BLOCK_LOW_AND_ABOVE"
                    )
                )
                
                if response and hasattr(response, 'generated_images') and response.generated_images:
                    return response.generated_images[0].image.image_bytes
                else:
                    last_error = "Safety filters blocked the generation or no image was returned"
            except Exception as e:
                last_error = str(e)
                continue
        
        raise Exception(f"Image generation failed: {last_error}")
    
    def _refine_prompt(self, prompt: str) -> str:
        try:
            refine_chat = self.client.chats.create(model="gemini-2.5-flash-lite")
            refine_res = refine_chat.send_message(
                "You are an artist's prompt engineer. Create a highly detailed, "
                "cinematic physical description of the following subject. "
                "IMPORTANT: Remove all names of real people, politicians, or celebrities. "
                "Describe their facial features, hair, clothing, and the lighting style "
                f"generically so an artist can paint it without knowing who it is: '{prompt}'"
            )
            if refine_res.text:
                return refine_res.text
        except Exception:
            pass
        
        return prompt
    
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
