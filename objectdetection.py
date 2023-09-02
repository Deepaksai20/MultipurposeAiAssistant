import speech_recognition as sr
import pyttsx3
import pywhatkit
import os
import cv2
import requests
import cvlib as cv
from bs4 import BeautifulSoup
import re
import bardapi
import numpy as np
import face_recognition
from keras.preprocessing import image
from keras.models import model_from_json

# Initialize speech recognition and text-to-speech engines
listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Load the emotion detection model
model = model_from_json(open("models/emotion_model.json", "r").read())
model.load_weights('models/emotion_model_weights.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Text-to-speech function
def talk(text):
    engine.say(text)
    engine.runAndWait()

# Speech recognition function
def take_command():
    command = ""
    try:
        with sr.Microphone() as source:
            print('Listening...')
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
    except sr.UnknownValueError:
        talk("Please tell me something.")
    except sr.RequestError:
        talk("Sorry, I'm having trouble connecting to the speech recognition service.")
    return command

# Function to run the interactive assistant
def run_assistant():
    talk("Hello! I'm your interactive assistant. How can I assist you today?")
    while True:
        try:
            command = take_command()
            if command:
                print("User Command:", command)
                if 'search' in command:
                    search_query = command.split('search', 1)[1].strip()
                    talk('Searching for ' + search_query + ' on Google')
                    pywhatkit.search(search_query)
                elif 'open google chrome' in command:
                    open_google_chrome(command)
                elif 'object detection' in command:
                    perform_object_detection()
                elif 'perform emotion detection' in command:
                    talk('Sure, please hold a steady face in front of the camera.')
                    perform_emotion_detection()
                else:
                    chat_with_assistant(command)
        except KeyboardInterrupt:
            talk("Assistant terminated by user.")
            break

# Function to open Google Chrome
def open_google_chrome(command):
    try:
        if 'search' in command:
            search_query = command.split('search', 1)[1].strip()
            talk('Okay, opening Google Chrome and searching for ' + search_query)
            os.system(f'start chrome "https://www.google.com/search?q={search_query}"')
        else:
            talk('Sure, I will open Google Chrome for you.')
            os.system('start chrome')
    except Exception as e:
        talk('Oops! There was an error while trying to open Google Chrome.')

# Function to perform object detection and search for object price
def perform_object_detection():
    try:
        talk('Sure, please hold an object in front of the camera.')
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            bbox, label, conf = cv.detect_common_objects(frame)
            if 'person' in label:
                label.remove('person')
            detected_object = None
            for l, c in zip(label, conf):
                if c > 0.5:
                    detected_object = l
                    break
            if detected_object:
                search_object_price(detected_object)
                break
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        talk('Oops! There was an error during object detection.')

# Function to search for object price
def search_object_price(detected_object):
    search_query = f"{detected_object} price"
    search_results = pywhatkit.search(search_query)
    price = extract_price_from_results(search_results)
    if price:
        talk(f"The price of the {detected_object} is approximately {price}.")
    else:
        talk(f"I'm sorry, I couldn't find the price of the {detected_object}.")

# Function to extract price from search results
def extract_price_from_results(search_results):
    try:
        soup = BeautifulSoup(search_results, 'html.parser')
        prices = []

        # Find all elements that potentially contain prices
        price_elements = soup.find_all(text=re.compile(r'\$\d+\.\d{2}'))

        for element in price_elements:
            # Extract the price using regular expression
            price_match = re.search(r'\$\d+\.\d{2}', element)
            if price_match:
                prices.append(price_match.group())

        if prices:
            return ', '.join(prices)  # Combine multiple prices into a single string
        else:
            return None

    except Exception as e:
        print(e)
        return None

# Function to perform emotion detection
def perform_emotion_detection():
    try:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_recognition.face_locations(frame)

            for (top, right, bottom, left) in faces:
                face = gray[top:bottom, left:right]
                face = cv2.resize(face, (48, 48))
                face = np.expand_dims(face, axis=0)
                face = face / 255.0

                emotion_prob = model.predict(face)[0]
                predicted_emotion = emotion_labels[np.argmax(emotion_prob)]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, predicted_emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                talk(f"I detected a human face. You seem to be feeling {predicted_emotion}.")
                break

            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        talk('Oops! There was an error during emotion detection.')

# Function to chat with the assistant using Bard API
def chat_with_assistant(command):
    chat_prompt = f"You: {command}\nAssistant:"
    response = bardapi.core.Bard(token).get_answer(chat_prompt)
    talk(response['content'])

# Token for Bard API (replace with your token)
token = 'aQj-q4JBqFeP7eK2oMeMbPBVQlZrhrmbxayxj4irsy9ByPfyl6H8IkBaKbP8l8z0q7tLWQ.'

# Run the assistant
try:
    run_assistant()
except KeyboardInterrupt:
    talk("Assistant terminated by user.")
