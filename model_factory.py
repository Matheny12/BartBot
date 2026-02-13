from ai_models import AIModel
from gemini_model import GeminiModel
import streamlit as st

def get_model(model_type: str = "gemini") -> AIModel:
    if model_type == "gemini":
        api_key = st.secrets.get("GEMINI_KEY")
        return GeminiModel(api_key=api_key, bot_name="Bartholemew")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
