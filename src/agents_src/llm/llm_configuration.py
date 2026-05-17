#LLm configuration mapping agent names to LLms
from dotenv import load_dotenv
load_dotenv()
LLM_CONFIG = {
    "Questioning Answer Agent": {
        "model":"llama3-8b-8192",
        "temperature": 0.0,
    }
}