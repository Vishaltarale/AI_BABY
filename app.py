from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import speech_recognition as sr
import google.generativeai as genai
import uvicorn
import io
import logging

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
            return JSONResponse({
                "userText": "Audio too short",
                "aiResponse": "I didn't catch that. Could you speak a bit longer?"
            })

        # Use in-memory buffer instead of file I/O
        audio_stream = io.BytesIO(audio_data)

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 500
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.5

        user_text = ""
        try:
            with sr.AudioFile(audio_stream) as source:
                recognizer.adjust_for_ambient_noise(source, duration= 10)
                audio = recognizer.record(source)
                user_text = recognizer.recognize_google(audio, language="en-US")
                logger.info(f"Recognized speech: {user_text}")
        except sr.UnknownValueError:
            return JSONResponse({
                "userText": "Speech not recognized",
                "aiResponse": "Could you repeat that please?"
            })
        except sr.RequestError as e:
            logger.error(f"Google Speech API error: {str(e)}")
            return JSONResponse({
                "userText": "Error",
                "aiResponse": "I'm having trouble connecting. Try again!"
            })

        # AI response from Gemini
        try:
            prompt = f"You are a friendly French-AI assistant for a baby. Respond simply and engagingly to: '{user_text}'. Keep it very short and friendly."
            response = model.generate_content(prompt, generation_config={"max_output_tokens": 60})
            ai_response = response.text
            logger.info(f"AI response: {ai_response}")
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            ai_response = "I'm thinking slowly today. Please ask again!"

        return JSONResponse({
            "userText": user_text,
            "aiResponse": ai_response
        })

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return JSONResponse({
            "userText": "Error",
            "aiResponse": "Oops! Something went wrong. Try again!"
        })

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
