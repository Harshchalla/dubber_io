import dataclasses
import asyncio
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


def preprocess_clip(video_filepath: str) -> str:
    basename = '.'.join(os.path.basename(video_filepath).split('.')[:-1])
    video = VideoFileClip(video_filepath)
    audio_filepath = basename+'.wav'
    video.audio.write_audiofile(audio_filepath)
    video.close()
    return audio_filepath


def audio_transcription(input_audio_clip):
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

async def translate_audio(text, source_audiopath: str, target_language: str, num: int):
    TARGET_LANGUAGE_TO_API_VALUE_MAP = {
        'English': 'en', 
        'French': 'fr',
        'Spanish': 'es',
        'German': 'de', 
    }
    target_language = TARGET_LANGUAGE_TO_API_VALUE_MAP[target_language]


    output = await replicate_audio_translation(text, source_audiopath, target_language)
    res = requests.get(output)
    if res.status_code != 200:
        print('getting output failed.')
        return 
    translated_output_file = '.'.join(os.path.basename(source_audiopath).split('.')[:-1]) + f'_translated_to_{target_language}_{num}.wav'
    with open(translated_output_file, 'wb') as f: f.write(res.content)
    return translated_output_file

async def replicate_audio_translation(text, source_audiopath, target_language):
    client = replicate.Client(os.environ['REPLICATE_API_TOKEN'])
    return await asyncio.to_thread(replicate.run, 
        "lucataco/xtts-v2:684bc3855b37866c0c65add2ff39c78f3dea3f4ff103a436465326e0f438d55e",
        input={
            "text": text,
            "speaker": open(source_audiopath, 'rb'),
            "language": target_language,
            "cleanup_voice": True
        }
    )

async def sync_lips(audio_filepath: str, video_filepath: str, num: int, start_ts: int, end_ts: int):
    from moviepy.editor import VideoFileClip, AudioFileClip
    video = VideoFileClip(video_filepath)#.subclip(start_ts, end_ts)
    # get the first frame of video and save it as png <vide_basename>_first_frame.png
    from PIL import Image
    import os

    first_frame_path = f"{os.path.splitext(video_filepath)[0]}_first_frame_{num}.png"
    video.save_frame(first_frame_path, t=0)
    face_images = detect_and_crop_faces(first_frame_path, padding_percent=0.8)
    print(face_images)
    if not len(face_images):
        print('No face images found')
        return None

    inp = dict(audio=open(audio_filepath, 'rb'), image=open(face_images[0], 'rb'))
    output = await replicate_sync_lips(inp)
    print(output)

    res = requests.get(output['video'])
    if res.status_code != 200:
        print('Retrieving video failed')
        return None

    final_video_chunk_path = '.'.join(os.path.basename(audio_filepath).split('.')[:-1]) + f'_video_gen_{num}.mp4'
    with open(final_video_chunk_path, 'wb') as f: f.write(res.content)

    VideoFileClip(final_video_chunk_path).set_audio(AudioFileClip(audio_filepath)).write_videofile(final_video_chunk_path)
    return final_video_chunk_path

async def replicate_sync_lips(inp):
    client = replicate.Client(os.environ['REPLICATE_API_TOKEN'])
    return await asyncio.to_thread(replicate.run,
        "cjwbw/aniportrait-audio2vid:3f976d8f2308f5c676a484e873f7d1ac09763f789fa211894df1ed96d3d17cb2",
        input=inp
    )

import cv2
import numpy as np

def detect_and_crop_faces(image_path, padding_percent):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier('pretrained_haarcascades/haarcascade_frontalface_default.xml')

    # Read the input image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    cropped_faces = []

    for (x, y, w, h) in faces[:1]:  # only need the first image
        # Calculate padding
        padding = int(max(w, h) * padding_percent)
        
        # Calculate the square crop dimensions
        size = max(w, h) + 2 * padding
        
        # Calculate crop coordinates
        center_x = x + w // 2
        center_y = y + h // 2
        
        left = max(0, center_x - size // 2)
        top = max(0, center_y - size // 2)
        right = min(image.shape[1], left + size)
        bottom = min(image.shape[0], top + size)
        
        # Crop the face
        face_crop = image[top:bottom, left:right]
        
        # Resize to ensure square shape
        face_crop = cv2.resize(face_crop, (size, size))
        
        cropped_faces.append(face_crop)

    # save the cropped images as <image_path>_face.png
    ret = []
    if cropped_faces:
        for i, face in enumerate(cropped_faces):
            ret.append(f"{os.path.splitext(image_path)[0]}_face_{i}.png")
            cv2.imwrite(ret[-1], face)
    return ret


def merge_transcriptions(transcriptions: list[UnitTranscription], min_gap: float):
    """
    Merge transcriptions to have a delta in between range(min_gap, max_gap) seconds
    """
    merged_transcriptions = []
    current_transcription = None

    for transcription in transcriptions:
        if current_transcription is None:
            current_transcription = transcription
        else:
            gap = transcription.end_timestamp - current_transcription.start_timestamp
            if gap > min_gap:
                merged_transcriptions.append(current_transcription)
                current_transcription = transcription
            else:
                current_transcription.end_timestamp = transcription.end_timestamp
                current_transcription.content += " " + transcription.content

    if current_transcription:
        merged_transcriptions.append(current_transcription)

    return merged_transcriptions


async def main(video_filepath: str, source_language: str, target_language: str):
    audio_filepath = preprocess_clip(video_filepath)
    transcriptions = audio_transcription(audio_filepath)
    merged_transcriptions = merge_transcriptions(transcriptions, 10.0)

    print("\nTranslated Transcriptions:")
    translated_transcriptions: list[UnitTranscription] = []
    for transcription in merged_transcriptions:
        translated_text = get_translation(transcription.content, source_language=source_language, target_language=target_language)
        print(f"[{transcription.start_timestamp:.2f} - {transcription.end_timestamp:.2f}] {translated_text['text_output']}")
        translated_transcriptions.append(UnitTranscription(transcription.start_timestamp, transcription.end_timestamp, translated_text['text_output']))

    print('Translating audio ...')
    translated_audiopath = await translate_audio(' '.join([x.content for x in translated_transcriptions]), audio_filepath, target_language, 0)

    from moviepy.editor import VideoFileClip, AudioFileClip
    import uuid
    final_videopath = f'{uuid.uuid4()}.mp4'
    VideoFileClip(video_filepath).set_audio(AudioFileClip(translated_audiopath)).write_videofile(final_videopath)
    print(final_videopath)


    # TODO: (rohan) fix lip sync and use below code
    return
    tasks = [translate_audio(transcription.content, audio_filepath, target_language, i) for i, transcription in enumerate(translated_transcriptions)]
    translated_audio_files = await asyncio.gather(*tasks)
    
    print('Syncing lips ...')
    tasks = [sync_lips(audio_filepath, video_filepath, num, unit_transcription.start_timestamp, unit_transcription.end_timestamp) for num, (audio_filepath, unit_transcription) in enumerate(zip(translated_audio_files, translated_transcriptions))]
    lip_sync_video_files = await asyncio.gather(*tasks)

    # combine all the lip sync videos
    from moviepy.editor import VideoFileClip, concatenate_videoclips

    clips = [VideoFileClip(file) for file in lip_sync_video_files if file is not None]
    final_video = concatenate_videoclips(clips)
    final_video.write_videofile(f'{os.path.splitext(video_filepath)[0]}_translated_{target_language}.mp4')
    return 

if __name__ == "__main__":
    #detect_and_crop_faces('input_video_first_frame.png', 0.8)
    #exit()
    #ret = sync_lips('input_video_translated_to_fr.wav', './input_video.mp4')
    #if ret == None: print('Failed')
    #exit()
    import time
    start = time.monotonic()
    asyncio.run(main('./input_video_trimmed.mp4', 'English', 'French'))
    end = time.monotonic()
    print('Total time taken:', end-start, 'secs')
    exit()

    text = '''Los demócratas lo saben y yo no lo he leído porque no sé de qué se trata. Es más fácil que decir que lo leí y sabes todas las demás cosas. No, no he leído ninguno, y hay cosas que no a todos les gustaría, pero hay algunas que no tienen nada que ver conmigo.'''
    source_audiopath = './tmp/trump_trimmed.mp3'
    translate_audio(text, source_audiopath, 'es')

