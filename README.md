# Video-to-Video Translation Service

This project is a FastAPI-based web application that processes videos by extracting audio, transcribing it, translating the transcriptions, and providing a translated version of the video. It integrates with several key technologies including:

- **FastAPI**: For building the API.
- **Nginx**: For reverse proxy and server configuration.
- **AWS**: For storage and deployment.
- **Whisper**: For speech-to-text transcription.
- **Replicate Seamless Communication API**: For text-to-text translation.
- **Anipoartier**: For future functionality or integration (this is a placeholder for additional APIs or functionality).

## Key Features

- **Video Upload**: Users can upload videos for processing.
- **Audio Extraction**: The system extracts audio from the uploaded video.
- **Transcription**: Transcribes audio using Whisper.
- **Translation**: Translates the transcribed text using the Seamless Communication API.
- **JSON Response**: Provides a structured JSON response with timestamps, original transcriptions, and translated text.
  
## Technologies Used

- **FastAPI**: Backend web framework.
- **Nginx**: As a reverse proxy server for load balancing and serving the API.
- **AWS (S3, EC2)**: For file storage and deployment.
- **Whisper**: For transcription services (speech-to-text).
- **Replicate API (Seamless Communication)**: For text-to-text translation.
- **Anipoartier**: Placeholder API for further enhancements.

## Requirements

1. Python 3.8+
2. FastAPI
3. Whisper (OpenAI's model for transcription)
4. Replicate Python client
5. MoviePy (for video processing)
6. Pydantic (for data validation)
7. AWS account (for deployment)

