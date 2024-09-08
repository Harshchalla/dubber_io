import replicate
import requests
from moviepy.editor import VideoFileClip


def lipsync(translated_audio_path: str, video_path: str) -> str:
    pass


def download_file(file_url, local_filename):
    response = requests.get(file_url, stream=True)
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded successfully as {local_filename}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
    return local_filename


def preprocess_clip(input_clip="input_video.mp4"):
    video = VideoFileClip(input_clip)
    audio = video.audio
    audio.write_audiofile("input_audio.mp3")
    audio.close()
    video.close()


def speech_to_speech_translation(input_audio_file, lang):
    output = replicate.run(
        "cjwbw/seamless_communication:668a4fec05a887143e5fe8d45df25ec4c794dd43169b9a11562309b2d45873b0",
        input={
            "task_name": "S2ST (Speech to Speech translation)",
            "input_audio": open(input_audio_file, "rb"),  # Replace with your URI
            "input_text_language": "None",
            "max_input_audio_length": 60,
            "target_language_text_only": "Norwegian Nynorsk",
            "target_language_with_speech": lang
        }
    )
    print(output)
    return output["audio_output"]


def main(lang: str) -> str:
    preprocess_clip()
    audio_output_filepath = speech_to_speech_translation("input_audio.mp3", lang)
    download_file(audio_output_filepath, "output_audio.wav")
    # lipsync
    pass


if __name__ == "__main__":
    main(lang="Japanese")
