import streamlit as st
import os
import torch
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from utils import get_answer, text_to_speech, autoplay_audio, speech_to_text
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *  # Ensure this import is correct for your float functionality

# Float feature initialization
float_init()

st.sidebar.title("Language Settings")
language = st.sidebar.selectbox("Choose language", ["English", "Hindi"])

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "🍽️ Welcome to TasteMate! 🍽️\n\nHi there! We're excited to help you discover delicious cuisines and enhance your dining experience. How can we assist you today? 🌟"}
        ]

initialize_session_state()

st.title("🍔TasteMate Conversational bot🍕")

# Create footer container for the microphone
footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if audio_bytes:
    with st.spinner("Transcribing..."):
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)

        transcript = speech_to_text(webm_file_path, language)
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            with st.chat_message("user"):
                st.write(transcript)
            os.remove(webm_file_path)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking🤔..."):
            # Query the LLaMA-based RAG model
            documents = SimpleDirectoryReader("/home/aiml/Abir/Useful_CODE/llm/submit_proposal_cap1/data/").load_data()
            system_prompt = """You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided no unwanted answers."""
            query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
            llm = HuggingFaceLLM(
                context_window=4096,
                max_new_tokens=256,
                generate_kwargs={"temperature": 0.1, "do_sample": False},
                system_prompt=system_prompt,
                query_wrapper_prompt=query_wrapper_prompt,
                tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
                model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
                device_map="auto",
                model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
            )
            embed_model = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
            service_context = ServiceContext.from_defaults(
                chunk_size=1024,
                llm=llm,
                embed_model=embed_model
            )
            index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            query_engine = index.as_query_engine()
            response_obj = query_engine.query(st.session_state.messages[-1]["content"])

            # Extract response text from the response object
            final_response = response_obj.response  # Assuming 'response' is the attribute containing text

        with st.spinner("Generating audio response..."):    
            audio_file = text_to_speech(final_response, language)
            if audio_file:
                autoplay_audio(audio_file)
                os.remove(audio_file)  # Clean up the temporary file
            else:
                st.error("Failed to generate audio response.")
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})


# Float the footer container and provide CSS to target it with
footer_container.float("bottom: 0rem;")
