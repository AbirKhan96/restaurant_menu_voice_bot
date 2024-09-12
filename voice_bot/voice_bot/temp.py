

# from openai import OpenAI
# import os
# from dotenv import load_dotenv
# import base64
# import streamlit as st
# import torch
# from llama_index import VectorStoreIndex, ServiceContext
# from llama_index.llms import HuggingFaceLLM
# from llama_index.prompts.prompts import SimpleInputPrompt
# #from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from llama_index.embeddings import LangchainEmbedding
# from llama_index.readers import SimpleDirectoryReader 
# load_dotenv()
# api_key = os.getenv("openai_api_key")

# client = OpenAI(api_key=api_key)

# language= 'English'
# input_text= "what is the price of Chicken biryani"
# voice = "nova" if language == "English" else "onyx"
# response = client.audio.speech.create(
#         model="tts-1",
#         voice=voice,
#         input=input_text
#     )

# test.py

import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import tempfile
import base64

load_dotenv()
api_key = os.getenv("openai_api_key")

client = OpenAI(api_key=api_key)

def text_to_speech(input_text, language):
    try:
        voice = "nova" if language == "English" else "onyx"

        # Use the correct method and handle response correctly
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=input_text
        )
        
        # Read binary content from response
        audio_content = response.content  # .content for binary response
        
        # Save audio to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with open(temp_file.name, "wb") as audio_file:
            audio_file.write(audio_content)
        
        return temp_file.name
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    audio_html = f"""
    <audio controls autoplay>
    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# Streamlit app
st.title("Text-to-Speech with OpenAI")

language = st.selectbox("Choose language", ["English", "Hindi"])
input_text = st.text_input("Enter text")

if st.button("Generate Speech"):
    if input_text:
        with st.spinner("Generating audio..."):
            audio_file = text_to_speech(input_text, language)
            if audio_file:
                st.success("Audio generated successfully!")
                autoplay_audio(audio_file)
                os.remove(audio_file)  # Clean up the temporary file
    else:
        st.warning("Please enter some text.")
