import dataclasses
import requests
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
import replicate
import os


@dataclasses.dataclass
class UnitTranscription:
    start_timestamp: float
    end_timestamp: float
    content: str


def preprocess_clip(input_clip="input_video.mp4"):
    video = VideoFileClip(input_clip)
    audio = video.audio
    audio.write_audiofile("input_audio.mp3")
    audio.close()
    video.close()


def audio_transcription(input_audio_clip="input_audio.mp3"):
    model = whisper.load_model("base")
    result = model.transcribe(input_audio_clip)

    # Extract segments with timestamps and text
    segments = result['segments']
    transcriptions = []
    # Print the transcript with timestamps
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        transcriptions.append(UnitTranscription(start_time, end_time, text))
        print(f"[{start_time:.2f} - {end_time:.2f}] {text}")
    return transcriptions
    # with open("transcript_with_timestamps.txt", "w") as f:
    #     for segment in segments:
    #         start_time = segment['start']
    #         end_time = segment['end']
    #         text = segment['text']
    #         f.write(f"[{start_time:.2f} - {end_time:.2f}] {text}\n")


def get_translation(text, source_language = "English", target_language= "French"):
    api_token = os.getenv("REPLICATE_API_TOKEN")

    if not api_token:
        raise EnvironmentError("API token not found. Please set the REPLICATE_API_TOKEN environment variable.")

    client = replicate.Client(api_token)

    output = client.run(
        "cjwbw/seamless_communication:668a4fec05a887143e5fe8d45df25ec4c794dd43169b9a11562309b2d45873b0",
        input={
            "task_name": "T2TT (Text to Text translation)",
            "input_text": text,
            "input_text_language": source_language,  # Specify input language here
            "target_language_text_only": target_language  # Target language for translation
        }
    )

    return output

def main():
    preprocess_clip()
    transcriptions = audio_transcription()
    print("\nTranslated Transcriptions:")
    for transcription in transcriptions:
        translated_text = get_translation(transcription.content, source_language="English", target_language="French")
        print(f"[{transcription.start_timestamp:.2f} - {transcription.end_timestamp:.2f}] {translated_text}")





if __name__ == "__main__":
    main()
