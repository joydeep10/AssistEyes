import os
import base64
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import re
from PIL import Image, ImageDraw
from io import BytesIO
from gtts import gTTS, lang
import pygame
from google.cloud import vision
import pytesseract
import time
from googletrans import Translator

# Configure Tesseract-OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\joyde\AppData\Local\Programs\Tesseract-OCR\tesseract'

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not GOOGLE_API_KEY or not GOOGLE_APPLICATION_CREDENTIALS:
    st.error("Google API Key or Application Credentials are not set. Please check your environment variables.")
    st.stop()

# Set Google Vision API credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY)

# Initialize Pygame Mixer once globally
pygame.mixer.init()

# Function to process the input as a local file or URL
def process_image_input(image_input):
    if os.path.isfile(image_input):
        with open(image_input, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded_image}"
    elif image_input.startswith("http://") or image_input.startswith("https://"):
        response = requests.get(image_input)
        try:
            img = Image.open(BytesIO(response.content))
            img.verify()
            return image_input
        except Exception:
            raise ValueError("Invalid or inaccessible image URL.")
    else:
        raise ValueError("Input must be a valid file path or URL.")

# Function for OCR text extraction
def extract_text(image_input):
    if os.path.isfile(image_input):
        img = Image.open(image_input).convert("L")
    else:
        response = requests.get(image_input)
        img = Image.open(BytesIO(response.content)).convert("L")
    return pytesseract.image_to_string(img).strip()

# Function to translate text
def translate_text(text, target_language):
    translator = Translator()
    translated = translator.translate(text, src='en', dest=target_language)
    return translated.text

# Function to generate and play audio using context-specific files
def play_audio(text, lang_code="en", context="general"):
    translated_text = translate_text(text, lang_code) if lang_code != 'en' else text
    output_file = f"output_audio_{context}_{int(time.time())}.mp3"
    tts = gTTS(text=translated_text, lang=lang_code)
    tts.save(output_file)
    pygame.mixer.music.load(output_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass
    pygame.mixer.music.stop()
    os.remove(output_file)
    return translated_text

# Function to stop audio playback
def stop_audio():
    pygame.mixer.music.stop()

# Function for object detection using Google Vision API
def detect_and_draw_objects(image_input):
    client = vision.ImageAnnotatorClient()
    if os.path.isfile(image_input):
        with open(image_input, "rb") as image_file:
            content = image_file.read()
    else:
        response = requests.get(image_input)
        content = response.content

    vision_image = vision.Image(content=content)
    objects = client.object_localization(image=vision_image).localized_object_annotations

    if objects:
        img = Image.open(BytesIO(content))
        draw = ImageDraw.Draw(img)
        for obj in objects:
            box = [(vertex.x * img.width, vertex.y * img.height) for vertex in obj.bounding_poly.normalized_vertices]
            draw.line(box + [box[0]], width=3, fill="#00FF00")
            draw.text((box[0][0], box[0][1] - 10), f"{obj.name} ({obj.score:.2%})", fill="#00FF00")
        return img, objects
    else:
        return None, None

# Streamlit App setup
st.title("Visual Assistance for the Impaired")
available_languages = lang.tts_langs()
language_code = st.selectbox("Select Language for Audio Playback:", options=list(available_languages.keys()), format_func=lambda x: f"{available_languages[x]} ({x})")
selected_language_code = language_code.split('(')[-1].strip(')')

image_input = st.text_input("Enter the image URL or local file path:")
col1, col2, col3 = st.columns(3)
with col1:
    describe_button = st.button("Describe Scene with Playback")
with col2:
    ocr_button = st.button("Extract Text and Play Audio")
with col3:
    detect_button = st.button("Detect Object/Objects")

stop_audio_button = st.button("Stop Audio", key="stop_audio_button")

# Handling button actions
if describe_button and image_input:
    try:
        processed_image = process_image_input(image_input)
        st.image(processed_image, caption="Uploaded Image", use_container_width=True)
        message = HumanMessage(content=[{"type": "text", "text": "Describe this image:"}, {"type": "image_url", "image_url": processed_image}])
        response = llm.invoke([message])
        response_content = re.sub(r"\*\*(.*?)\*\*", r"\1", response.content)
        st.write(f"### Image Description in {language_code}:")
        translated = translate_text(response_content, selected_language_code)
        st.write(translated)
        play_audio(response_content, lang_code=selected_language_code, context='scene')
    except Exception as e:
        st.error(f"Error: {e}")

if ocr_button and image_input:
    try:
        extracted_text = extract_text(image_input)
        if extracted_text:
            st.write("### Extracted Text:")
            translated_text = play_audio(extracted_text, lang_code=selected_language_code, context='ocr')
            st.write(f"### Translated Text: {translated_text}")
        else:
            st.warning("No text found in the image.")
    except Exception as e:
        st.error(f"Error: {e}")

if detect_button and image_input:
    try:
        img_with_boxes, detected_objects = detect_and_draw_objects(image_input)
        if detected_objects:
            st.image(img_with_boxes, caption="Detected Objects with Bounding Boxes", use_column_width=True)
            st.write("### Detected Objects:")
            for obj in detected_objects:
                st.write(f"{obj.name} (confidence: {obj.score:.2%})")
        else:
            st.warning("No objects detected in the image.")
    except Exception as e:
        st.error(f"Error: {e}")

if stop_audio_button:
    stop_audio()
