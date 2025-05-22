from fastapi import FastAPI, Request, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import speech_recognition as sr
import google.generativeai as genai
import uvicorn
import tempfile
import os
import io
import wave
import logging
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Gemini AI
genai.configure(api_key='AIzaSyDQ7alBZulcqJbFI1RvKmWpMm6rsSQi2Ec')
model = genai.GenerativeModel('gemini-1.5-flash')

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-audio")
async def process_audio(request: Request):
    try:
        # Get the audio data from the request
        audio_data = await request.body()
        logger.info(f"Received audio data of size: {len(audio_data)} bytes")
        
        if len(audio_data) < 100:
            logger.warning("Audio data too small, likely invalid")
            return JSONResponse({
                "userText": "Audio too short",
                "aiResponse": "I didn't catch that. Could you speak a bit longer?"
            })
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            
        # Save the audio data to the temporary file
        with open(temp_path, 'wb') as f:
            f.write(audio_data)
            
        logger.info(f"Saved audio to temporary file: {temp_path}")
        
        # Use multiple speech recognition engines for better results
        user_text = ""
        recognizer = sr.Recognizer()
        
        # Configure the recognizer for better results
        recognizer.energy_threshold = 500  # Lower energy threshold for quieter audio
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8  # Shorter pause threshold for natural speech
        
        try:
            with sr.AudioFile(temp_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record audio from the file
                audio = recognizer.record(source)
                
                # Try Google's speech recognition with language specification
                try:
                    user_text = recognizer.recognize_google(audio, language="en-US")
                    logger.info(f"Google recognized: {user_text}")
                except sr.UnknownValueError:
                    logger.warning("Google couldn't understand audio")
                    # Try Sphinx as fallback
                    try:
                        user_text = recognizer.recognize_sphinx(audio)
                        logger.info(f"Sphinx recognized: {user_text}")
                    except:
                        logger.warning("Sphinx recognition failed or not available")
                except sr.RequestError as e:
                    logger.error(f"Google speech API error: {str(e)}")
                    # Try Sphinx if Google API fails
                    try:
                        user_text = recognizer.recognize_sphinx(audio)
                        logger.info(f"Sphinx recognized (after Google API error): {user_text}")
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
            logger.info(f"Deleted temporary file: {temp_path}")
        except Exception as e:
            logger.error(f"Error deleting temporary file: {str(e)}")
        
        # If we couldn't recognize anything
        if not user_text:
            logger.warning("Speech recognition failed with all engines")
            return JSONResponse({
                "userText": "Speech not recognized",
                "aiResponse": "I'm sorry, I couldn't understand what you said. Could you try speaking more clearly or in a quieter place?"
            })
        
        # Get AI response
        try:
            prompt = f"You are a friendly AI assistant for a baby. Detect the language of this text: '{user_text}', and respond in that same language in a simple, short, and engaging way that a baby would enjoy make is shorter in 2-3 sent."
            response = model.generate_content(prompt)
            ai_response = response.text
            logger.info(f"AI response: {ai_response}")
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            ai_response = "I'm having trouble thinking right now. Can you ask me again?"
        
        # Return the response
        return JSONResponse({
            "userText": user_text,
            "aiResponse": ai_response
        })
            
    except Exception as e:
        logger.error(f"General error processing request: {str(e)}")
        return JSONResponse({
            "userText": "Error processing speech",
            "aiResponse": "Oops! Something went wrong. Please try again!"
        })

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)