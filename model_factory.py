from ai_models import AIModel
from GeminiBartModel import GeminiModel
from BartBotModel import BartBotModel
import streamlit as st

def get_model(model_type: str = "gemini") -> AIModel:
    if model_type == "GeminiBart":
        api_key = st.secrets.get("GEMINI_KEY")
        return GeminiModel(api_key=api_key, bot_name="Bartholemew")
    elif model_type == "BartBot":
        try:
            from BartBotModel import BartBotModel
            return BartBotModel("meta-llama/Llama-3.1-8B-Instruct")
        except Exception as e:
            st.error(f"Failed to load BartBot: {str(e)}")
            st.info("Going back to GeminiBart")
            api_key = st.secrets.get("GEMINI_KEY")
            return GeminiModel(api_key=api_key, bot_name="Bartholemew")

    else:
        api_key = st.secrets.get("GEMINI_KEY")
        return GeminiModel(api_key=api_key, bot_name="Bartholemew")