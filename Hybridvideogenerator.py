"""
Hybrid Video Generator - Works both locally and on cloud

Uses AnimateDiff when available (local with GPU), falls back to Replicate (cloud)
"""

import os
from typing import Optional
import base64
from io import BytesIO

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    class st:
        @staticmethod
        def secrets():
            return {}

from PIL import Image as PILImage
import requests

try:
    from AnimateDiff import AnimateDiffGenerator, SimpleAnimateDiff
    ANIMATEDIFF_AVAILABLE = True
except ImportError:
    ANIMATEDIFF_AVAILABLE = False
    print("[HybridVideoGen] AnimateDiff not available")

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    print("[HybridVideoGen] Replicate not available")

try:
    from SimpleVideoGenerator import UniversalVideoGenerator
    SIMPLE_VIDEO_AVAILABLE = True
except ImportError:
    SIMPLE_VIDEO_AVAILABLE = False
    print("[HybridVideoGen] SimpleVideoGenerator not available")


class HybridVideoGenerator:
    """
    Smart video generator that uses the best available method:
    1. Local AnimateDiff (if GPU available) - FREE & UNLIMITED
    2. Replicate API (if AnimateDiff unavailable) - PAID but works everywhere
    """
    
    def __init__(self):
        self.animatediff = None
        self.method = None
        self._detect_best_method()
    
    def _detect_best_method(self):
        """Automatically detect which method to use"""
        
        force_method = os.getenv("FORCE_VIDEO_METHOD", "").lower()
        if force_method == "local":
            self.method = "local"
            print("[HybridVideoGen] Forced to use LOCAL (AnimateDiff)")
            return
        elif force_method == "replicate":
            self.method = "replicate"
            print("[HybridVideoGen] Forced to use REPLICATE (cloud)")
            return
        elif force_method == "free":
            self.method = "free"
            print("[HybridVideoGen] Forced to use FREE methods")
            return
        
        if SIMPLE_VIDEO_AVAILABLE:
            self.method = "free"
            print("[HybridVideoGen] Using FREE video generation (HuggingFace + Fallback)")
            return
        
        if REPLICATE_AVAILABLE:
            self.method = "replicate"
            print("[HybridVideoGen] Using REPLICATE (requires API token)")
            return
        
        if ANIMATEDIFF_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    self.method = "local"
                    print("[HybridVideoGen] GPU detected! Using LOCAL AnimateDiff")
                    return
                else:
                    print("[HybridVideoGen] WARNING: No GPU - AnimateDiff will be VERY slow")
                    self.method = "local"
                    return
            except:
                pass
        
        self.method = None
        print("[HybridVideoGen] ERROR: No video generation method available!")
    
    def generate_video(self, image_data: bytes, prompt: str = None) -> bytes:
        """Generate video using best available method"""
        
        if self.method == "free":
            return self._generate_free(image_data, prompt)
        elif self.method == "local":
            return self._generate_local(image_data, prompt)
        elif self.method == "replicate":
            return self._generate_replicate(image_data, prompt)
        else:
            raise Exception(
                "No video generation method available!\n\n"
                "Install SimpleVideoGenerator.py for FREE video generation"
            )
    
    def _generate_free(self, image_data: bytes, prompt: str) -> bytes:
        """Generate using FREE methods (HuggingFace + Fallback)"""
        try:
            generator = UniversalVideoGenerator()
            return generator.generate_video(image_data, prompt)
        except Exception as e:
            print(f"[HybridVideoGen] Free generation failed: {e}")
            raise
    
    def _generate_local(self, image_data: bytes, prompt: str) -> bytes:
        """Generate using local AnimateDiff"""
        try:
            if self.animatediff is None:
                use_simple = os.getenv("USE_SIMPLE_ANIMATEDIFF", "true").lower() == "true"
                if use_simple:
                    self.animatediff = SimpleAnimateDiff()
                    print("Loaded SimpleAnimateDiff")
                else:
                    self.animatediff = AnimateDiffGenerator()
                    print("Loaded AnimateDiffGenerator")
            
            num_frames = int(os.getenv("ANIMATEDIFF_FRAMES", "16"))
            fps = int(os.getenv("ANIMATEDIFF_FPS", "8"))
            
            return self.animatediff.generate_from_image(
                image_data=image_data,
                prompt=prompt or "smooth motion",
                num_frames=num_frames,
                fps=fps
            )
        except Exception as e:
            print(f"Local generation failed: {e}")
            if REPLICATE_AVAILABLE:
                print("Falling back to Replicate...")
                self.method = "replicate"
                return self._generate_replicate(image_data, prompt)
            raise
    
    def _generate_replicate(self, image_data: bytes, prompt: str) -> bytes:
        """Generate using Replicate API"""
        try:
            api_token = None
            
            if STREAMLIT_AVAILABLE:
                try:
                    api_token = st.secrets["REPLICATE_API_TOKEN"]
                except (KeyError, AttributeError):
                    pass
            
            if not api_token:
                api_token = os.getenv("REPLICATE_API_TOKEN")
            
            if not api_token:
                raise ValueError(
                    "REPLICATE_API_TOKEN not found!\n\n"
                    "Add it to Streamlit secrets:\n"
                    "REPLICATE_API_TOKEN = \"r8_your_token_here\"\n\n"
                    "Get your token at: https://replicate.com/account/api-tokens"
                )
            
            os.environ["REPLICATE_API_TOKEN"] = api_token
            
            print(f"[Replicate] Token found: {api_token[:10]}...")
            
            img = PILImage.open(BytesIO(image_data))
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_uri = f"data:image/png;base64,{img_base64}"
            
            print("Calling Replicate API...")
            
            output = replicate.run(
                "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
                input={
                    "input_image": image_uri,
                    "cond_aug": 0.02,
                    "decoding_t": 7,
                    "video_length": "14_frames_with_svd",
                    "sizing_strategy": "maintain_aspect_ratio",
                    "motion_bucket_id": 127,
                    "frames_per_second": 7
                }
            )
            
            video_url = output if isinstance(output, str) else output[0] if isinstance(output, list) else str(output)
            video_response = requests.get(video_url, timeout=60)
            video_response.raise_for_status()
            
            return video_response.content
            
        except Exception as e:
            raise Exception(f"Replicate generation failed: {str(e)}")
    
    def get_method_info(self) -> dict:
        """Get information about current method"""
        return {
            "method": self.method,
            "animatediff_available": ANIMATEDIFF_AVAILABLE,
            "replicate_available": REPLICATE_AVAILABLE,
            "description": {
                "local": "LOCAL AnimateDiff (GPU) - Unlimited & Free",
                "replicate": "Replicate Cloud API - Paid (~$0.10/video)",
                None: "No method available - please configure"
            }.get(self.method)
        }