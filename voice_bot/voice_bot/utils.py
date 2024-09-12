from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
import streamlit as st
import tempfile
load_dotenv()
api_key = os.getenv("openai_api_key")

client = OpenAI(api_key=api_key)

def get_answer(messages, language):
    system_message = [{"role": "system", "content": f"You are a helpful bilingual AI chatbot. Respond in {language}."}]
    messages = system_message + messages
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    return response.choices[0].message.content

def speech_to_text(audio_data, language):
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file,
            language="en" if language == "English" else "hi"
        )
    return transcript

def text_to_speech(input_text, language):
    try:
        voice = "nova" if language == "English" else "onyx"

        # Use the correct method and handle response correctly
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=input_text
        )
        
        # Check if response has 'content' attribute and extract binary audio content
        if hasattr(response, 'content'):
            audio_content = response.content  # Ensure 'content' is the correct attribute
            
            # Save audio to a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with open(temp_file.name, "wb") as audio_file:
                audio_file.write(audio_content)
            
            return temp_file.name
        else:
            raise ValueError("Response does not contain 'content'.")
    except Exception as e:
        st.error(f"An error occurred during text-to-speech conversion: {e}")
        return None

def autoplay_audio(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        audio_html = f"""
        <audio controls autoplay>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred while playing audio: {e}")

