import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

from src.agents_src.llm.llm_configuration import LLM_CONFIG


def get_llm_for_agent(agent_name):

    model = LLM_CONFIG.get(agent_name, {}).get(
        "model",
        "groq/llama-3.3-70b-versatile"
    )

    temperature = LLM_CONFIG.get(agent_name, {}).get(
        "temperature",
        0.0
    )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    llm = LLM(
        model=model,
        temperature=temperature,
        api_key=api_key
    )

    return llm