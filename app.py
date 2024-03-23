import streamlit as st
import whisper
import openai
import os
from pytube import YouTube
from pathlib import Path

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = 'text-davinci-003'
WHISPER_MODEL = 'base'
YOUTUBE_VIDEO_URL = 'https://www.youtube.com/watch?v=GNd12j-CGeQ'
OUTPUT_AUDIO = Path(__file__).resolve().parent.joinpath('data', 'podcast.mp4')


def download_youtube_video(url, output_audio):
    youtube_video = YouTube(url)
    streams = youtube_video.streams.filter(only_audio=True)
    stream = streams.first()
    stream.download(filename=output_audio)


def summarize_text(transcript):
    system_prompt = "I would like for you to assume the role of a Life Coach"
    user_prompt = f"""Generate a concise summary of the text below.
    Text: {transcript}

    Add a title to the summary.

    Make sure your summary has useful and true information about the main points of the topic.
    Begin with a short introduction explaining the topic. If you can, use bullet points to list important details,
    and finish your summary with a concluding sentence"""
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo-16k',
                                            messages=[{
                                                'role': 'system',
                                                'content': system_prompt
                                            }, {
                                                'role': 'user',
                                                'content': user_prompt
                                            }],
                                            max_tokens=4096,
                                            temperature=1)
    summary = response['choices'][0]['message']['content']
    return summary


def main():
    st.title('YouTube Video Summarizer')
    st.sidebar.header('Input')
    video_url = st.sidebar.text_input('Enter YouTube Video URL', YOUTUBE_VIDEO_URL)

    if st.sidebar.button('Summarize'):
        st.sidebar.text('Downloading video...')
        download_youtube_video(video_url, OUTPUT_AUDIO)
        st.sidebar.text('Transcribing audio...')
        model = whisper.load_model(WHISPER_MODEL)
        transcript = model.transcribe(OUTPUT_AUDIO.as_posix())
        transcript_text = transcript['text']
        st.sidebar.text('Summarizing text...')
        summary = summarize_text(transcript_text)
        st.success('Summary generated successfully!')
        st.header('Summary')
        st.write(summary)


if __name__ == '__main__':
    main()
