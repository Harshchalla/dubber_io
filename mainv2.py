import dataclasses

import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip


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


def main():
    preprocess_clip()
    transcriptions = audio_transcription()
    print(transcriptions)


if __name__ == "__main__":
    main()
