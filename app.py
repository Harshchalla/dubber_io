import os
import dataclasses
import whisper
import replicate
from moviepy.video.io.VideoFileClip import VideoFileClip
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all origins and methods
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be changed to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all origins and methods
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be changed to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclasses.dataclass
class UnitTranscription:
    start_timestamp: float
    end_timestamp: float
    content: str

# Preprocess video to extract audio
def preprocess_clip(input_clip_path: str):
    video = VideoFileClip(input_clip_path)
    audio = video.audio
    audio_output_path = os.path.join(UPLOAD_DIR, "input_audio.mp3")
    audio.write_audiofile(audio_output_path)
    audio.close()
    video.close()
    return audio_output_path

# Transcribe audio using Whisper
def audio_transcription(input_audio_clip: str):
    model = whisper.load_model("base")
    result = model.transcribe(input_audio_clip)

    segments = result['segments']
    transcriptions = []
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        transcriptions.append(UnitTranscription(start_time, end_time, text))
    return transcriptions

# Translate text using Replicate API
def get_translation(text, source_language="English", target_language="French"):
    api_token = os.getenv("REPLICATE_API_TOKEN")

    if not api_token:
        raise EnvironmentError("API token not found. Please set the REPLICATE_API_TOKEN environment variable.")

    client = replicate.Client(api_token)

    output = client.run(
        "cjwbw/seamless_communication:668a4fec05a887143e5fe8d45df25ec4c794dd43169b9a11562309b2d45873b0",
        input={
            "task_name": "T2TT (Text to Text translation)",
            "input_text": text,
            "input_text_language": source_language,
            "target_language_text_only": target_language
        }
    )

    return output

# Model for response
class TranslationResponse(BaseModel):
    start_timestamp: float
    end_timestamp: float
    original_text: str
    translated_text: str

# Endpoint for video upload, transcription, and translation
@app.post("/process_video/")
async def process_video(
    video: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...)
):
    # Save the uploaded video
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Preprocess video (extract audio)
    audio_path = preprocess_clip(video_path)

    # Transcribe audio using Whisper
    transcriptions = audio_transcription(audio_path)

    # Translate transcriptions
    translated_segments = []
    for transcription in transcriptions:
        translated_text = get_translation(
            transcription.content, source_language, target_language
        )
        translated_segments.append(
            TranslationResponse(
                start_timestamp=transcription.start_timestamp,
                end_timestamp=transcription.end_timestamp,
                original_text=transcription.content,
                translated_text=translated_text['text_output']
            )
        )

    # Return the transcribed and translated data as JSON
    return JSONResponse(content=[segment.dict() for segment in translated_segments])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
