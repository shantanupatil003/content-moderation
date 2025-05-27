import re
from pydub import AudioSegment
import ffmpeg

def process_text(text):
    clean_text = re.sub(r'[^A-Za-z0-9 ]', '', text.lower())
    return clean_text


def process_audio(audio, time_stamps):

    #ffmpeg.input("sounds/quack.mp3").output("sounds/quack.wav")
    beep = AudioSegment.from_mp3("sounds/quack.wav")
    for start, end in time_stamps:
        audio = audio[:start] + beep + audio[end:]

    audio.export("results/processed_audio.wav", format="wav")
    

