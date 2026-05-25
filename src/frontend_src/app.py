import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.backend_src.main import settings
import streamlit as st
import requests
from src.frontend_src.config.fronted_settings import Settings

settings = Settings()

# Page config
st.set_page_config(
    page_title="AstraRAG",
    page_icon="🤖",
    layout="centered",
)
st.title("🚀 ASTRARAG - Agents RAG ChatBot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("role") == "assistant":
            sources = message.get("sources", [])
            tool_used = message.get("tool_used")
            rationale = message.get("rationale")
            if sources:
                st.markdown(f"**Sources:** {', '.join(sources)}")
            if tool_used or rationale:
                with st.expander("Show details (tool & rationale)"):
                    st.markdown(f"**Tool used:** {tool_used if tool_used else 'N/A'}")
                    st.markdown(f"**Rationale:** {rationale if rationale else 'N/A'}")
# User input
user_prompt = st.chat_input("Ask Chatbot...")
if user_prompt:
    # Show user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_prompt
    })

    payload = {
        "chat_history": st.session_state.chat_history
    }

    try:
        response = requests.post(settings.CHAT_ENDPOINT_URL, json=payload)
        response.raise_for_status()
        response_json = response.json()
        assistant_response = response_json.get("answer", "(No response)")
        tool_used = response_json.get("tool_used", "N/A")
        rationale = response_json.get("rationale", "N/A")
        sources = response_json.get("sources", [])
    except Exception as e:
        assistant_response = f"Error: {e}"
        tool_used = "N/A"
        rationale = "N/A"
        sources = []
    # Save assistant response
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_response,
        "tool_used": tool_used,
        "rationale": rationale,
        "sources": sources
    })
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
        if sources:
            st.markdown(f"**Sources:** {', '.join(sources)}")
        with st.expander("Show details (tools & rationale)"):
            st.markdown(f"**Tool Used:** {tool_used}")
            st.markdown(f"**Rationale:** {rationale}")