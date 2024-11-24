# Dictionary of personalized guidance for common objects
object_guidance = {
    "Bottle": "bottles that can be used to store or drink liquids.",
    "Chair": "chairs, which can be used for sitting.",
    "Car": "cars, which are vehicles used for transportation.",
    "Person": "people who might be nearby.",
    "Table": "tables, which can be used to place items or for eating.",
    "Book": "books, which you can read for information or leisure.",
    "Laptop": "laptops, which are portable computers often used for work or entertainment.",
    "Phone": "phones, which can be used for communication.",
    "Bag": "bags, which can be used to carry items.",
    "Cup": "cups, which can be used for drinking beverages.",
    "Plant": "plants, which add greenery to the surroundings.",
    "Dog": "dogs, which are common pets and loyal companions.",
    "Cat": "cats, which are common pets and often kept indoors.",
    "Shoe": "shoes, which are used for protecting and covering your feet.",
    "Pen": "pens, which can be used for writing.",
    "Bottle Cap": "bottle caps, which can seal or close bottles.",
    "Tire": "tires, which are circular rubber objects often part of vehicles.",
    "Wheel": "wheels, which are components used to enable movement in vehicles or objects.",
    "Fork": "forks, eating utensils used for picking up food.",
    "Knife": "knives, which can be used for cutting food or other items.",
    "Plate": "plates, which are used to serve or hold food.",
    "Glass": "glasses, used to hold and drink liquids.",
    "Keyboard": "keyboards, which are used to input text into computers.",
    "Monitor": "monitors, which are screens used to display content from a computer.",
}

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
from PIL import ImageFont


# Add a sidebar for the app
# Sidebar setup
st.sidebar.image("vision_logo.jpg", use_container_width=True)
st.sidebar.title("AssistEyes: Visual Guidance")
st.sidebar.write("Welcome to AssistEyes! A smart app designed to empower visually impaired individuals with real-time insights and assistance.")
st.sidebar.markdown("---")

# Features Section
st.sidebar.header("Features:")
st.sidebar.write("""
1. 🖼️ **Scene Description with Assistance**: Understand the surroundings with context-specific guidance and audio playback.
2. 📝 **Text Extraction**: Extract and translate text from images with audio feedback.
3. 🕵️ **Object Detection**: Identify objects, highlight them in the image, and get personalized guidance in audio.
4. 🌐 **Multi-Language Support**: Choose your preferred language for audio playback.
5. 🎵 **Audio Controls**: Stop ongoing audio playback anytime.
""")

# Personalized Assistance Section
st.sidebar.subheader("✨ Personalized Assistance:")
st.sidebar.write("""
- **What It Does**: Provides task-specific guidance based on the uploaded image. 
- **How We Achieve This**:
    - Detects objects in the image and identifies their purpose using label based object detection.
    - Adds detailed guidance about common objects, such as their functionality and context.
""")
st.sidebar.markdown("**Note:** We are using URLs for image input as they ensure faster processing")
# Tip Section
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tip**: Upload clear, high-resolution images for better results.")



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
# Function for OCR text extraction with formatting
def extract_text(image_input):
    if os.path.isfile(image_input):
        img = Image.open(image_input).convert("L")
    else:
        response = requests.get(image_input)
        img = Image.open(BytesIO(response.content)).convert("L")

    # Extract raw text
    raw_text = pytesseract.image_to_string(img).strip()

    # Step 1: Remove extra whitespaces and normalize text
    text = re.sub(r"\s+", " ", raw_text)  # Replace multiple spaces/newlines with a single space

    # Step 2: Organize into paragraphs
    paragraphs = text.split(". ")  # Split into sentences based on periods
    formatted_text = "\n\n".join(paragraph.strip() for paragraph in paragraphs if paragraph.strip())

    return formatted_text

# Function to translate text
def translate_text(text, target_language):
    translator = Translator()
    translated = translator.translate(text, src='en', dest=target_language)
    return translated.text

# Function to generate and play audio using unique output file
# Function to generate and play audio using unique output file
def play_audio(text, lang_code="en"):
    translated_text = translate_text(text, lang_code) if lang_code != 'en' else text
    if not translated_text:
        raise ValueError("Translated text is empty or invalid.")

    output_file = f"output_audio_{time.time()}.mp3"
    try:
        # Generate audio file
        tts = gTTS(text=translated_text, lang=lang_code)
        tts.save(output_file)

        # Play audio
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass
    except Exception as e:
        st.error(f"Error in playing audio: {e}")
    finally:
        # Stop and unload the audio to release the file handle
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()

        # Delete the temporary audio file
        try:
            os.remove(output_file)
        except Exception as delete_error:
            st.error(f"Error deleting audio file: {delete_error}")

    return translated_text


# Function to stop audio playback
def stop_audio():
    pygame.mixer.music.stop()

# Function for object detection using Google Vision API
# Function to detect objects, draw bounding boxes, and generate a description with counts
def detect_and_describe_objects_with_guidance(image_input):
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

        # Keep track of detected object names
        detected_object_names = []

        for obj in objects:
            object_name = obj.name
            detected_object_names.append(object_name)

            # Draw bounding box
            box = [(vertex.x * img.width, vertex.y * img.height) for vertex in obj.bounding_poly.normalized_vertices]
            draw.line(box + [box[0]], width=3, fill="#00FF00")
            draw.text((box[0][0], box[0][1] - 10), f"{object_name} ({obj.score:.2%})", fill="#00FF00")

        # Determine personalized guidance for each unique object
        unique_objects = set(detected_object_names)
        personalized_guidance = []

        for object_name in unique_objects:
            count = detected_object_names.count(object_name)
            # Use plural form if more than one of the object is detected
            if count > 1:
                plural_name = object_name + "s"  # Simple pluralization
            else:
                plural_name = object_name
            guidance = object_guidance.get(plural_name.rstrip("s"), f"{plural_name.lower()}, which is commonly seen.")
            personalized_guidance.append(guidance)

        # Combine personalized guidance into a single string
        combined_guidance = "The scene contains " + ", ".join(personalized_guidance)

        return img, objects, combined_guidance
    else:
        return None, None, "No objects detected in the scene."

# Streamlit App setup
st.title("Visual Assistance for the Impaired")
available_languages = lang.tts_langs()
language_code = st.selectbox("Select Language for Audio Playback:", options=list(available_languages.keys()), format_func=lambda x: f"{available_languages[x]} ({x})")
selected_language_code = language_code.split('(')[-1].strip(')')

image_input = st.text_input("Enter the image URL or local file path:")
col1, col2, col3 = st.columns(3)
with col1:
    describe_button = st.button("Describe Scene with Contextual Assistance")
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
        message = HumanMessage(content=[{"type": "text", "text": "You are an assistant for visually impaired individuals. First describe the scene in the Image and then provide detailed context specific guidance based on the Image. use numbers instead of bullet points"},
                                        {"type": "image_url", "image_url": processed_image}])
        response = llm.invoke([message])
        response_content = re.sub(r"\*\*(.*?)\*\*", r"\1", response.content)
        language_name = available_languages.get(language_code, "Unknown Language")
        st.write(f"### Image Description in {language_name}:")
        translated = translate_text(response_content, selected_language_code)
        st.write(translated)
        play_audio(response_content, lang_code=selected_language_code)
    except Exception as e:
        st.error(f"Error: {e}")

if ocr_button and image_input:
    try:
        # Extract text from the image
        extracted_text = extract_text(image_input)
        processed_image = process_image_input(image_input)
        st.image(processed_image, caption="Uploaded Image", use_container_width=True)
        if extracted_text:
            # Translate the extracted text to the selected language
            translated_text = translate_text(extracted_text, selected_language_code)

            # Display the translated text only
            st.write("### Extracted Text in Selected Language:")
            st.write(translated_text)

            # Play the translated text as audio
            play_audio(translated_text, lang_code=selected_language_code)
        else:
            no_text_message = "No text found in the image."
            st.warning(no_text_message)

            # Play the "No text found" message in audio
            play_audio(no_text_message, lang_code=selected_language_code)
    except Exception as e:
        st.error(f"Error: {e}")


# Update the detect_button action
if detect_button and image_input:
    try:
        # Detect objects and generate guidance
        img_with_boxes, detected_objects, combined_guidance = detect_and_describe_objects_with_guidance(image_input)

        if detected_objects:
            # Display the image with bounding boxes
            st.image(img_with_boxes, caption="Detected Objects with Bounding Boxes", use_container_width=True)

            # Display detected objects
            st.write("### Detected Objects:")
            for obj in detected_objects:
                st.write(f"{obj.name} (confidence: {obj.score:.2%})")

            # Translate the combined guidance into the selected language
            translated_guidance = translate_text(combined_guidance, selected_language_code)

            # Display translated guidance
            st.write("### Personalized Guidance in Selected Language:")
            st.write(translated_guidance)

            # Play the translated guidance in audio
            play_audio(translated_guidance, lang_code=selected_language_code)
        else:
            # Handle case when no objects are detected
            no_objects_message = "No objects detected in the image."
            translated_message = translate_text(no_objects_message, selected_language_code)

            # Display warning and play audio
            st.warning(translated_message)
            play_audio(translated_message, lang_code=selected_language_code)
    except Exception as e:
        st.error(f"Error: {e}")



if stop_audio_button:
    stop_audio()
