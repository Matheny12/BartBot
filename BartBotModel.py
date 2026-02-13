from ai_models import AIModel
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image
import io

class BartBotModel(AIModel):
    def __init__(self, model_path: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.image_model = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        self.vision_model = None

    def generate_response(self, messages: List[Dict], system_prompt: str, file_data: Optional[Dict] = None) -> str:
        prompt = self._format_messages(messages, system_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Bartholemew:")[-1].strip()
        return response

    def _format_messages(self, messages: List[Dict], system_prompt: str) -> str:
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")

        for msg in messages:
            role = "User" if msg["role"] == "user" else "Bartholemew"
            content = msg["content"]
            if not content.startswith("IMAGE_DATA:"):
                prompt_parts.append(f"{role}: {content}\n")

        prompt_parts.append("Bartholemew:")
        return "\n".join(prompt_parts)
    
    def generate_image(self, prompt: str) -> bytes:
        image = self.image_model(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    
    def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        if self.vision_model is None:
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.vision_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            self.vision_processor = processor

        image = Image.open(io.BytesIO(image_bytes))
        inputs = self.vision_processor(images=image, text=prompt, return_tensors="pt").to(self.vision_model.device)

        with torch.no_grad():
            outputs = self.vision_model.generate(**inputs, max_new_tokens=100)

        return self.vision_processor.decode(outputs[0], skip_special_tokens=True)