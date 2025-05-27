from config import auth_key
import os
import requests
import time
import pprint
import helpers
import numpy as np
import pandas as pd
from pydub import AudioSegment
import ffmpeg

base_url = "https://api.assemblyai.com"

headers = {"authorization": auth_key}

audio_file = "data/audio/BestFriendX.mp3"



#ffmpeg.input(audio_file).output("data/audio/audio_to_process.wav")

audio = AudioSegment.from_mp3("data/audio/audio_to_process.wav")

with open(audio_file, "rb") as f:
    response = requests.post(base_url + "/v2/upload", headers=headers, data=f)
if response.status_code != 200:
    print(f"Error: {response.status_code}, Response: {response.text}")
    response.raise_for_status()
upload_json =  response.json()
upload_url = upload_json["upload_url"]

endpoint = base_url + "/v2/transcript"

data = {
    "audio_url": upload_url,
    "content_safety": True
    }

response = requests.post(endpoint, headers=headers, json=data)

if response.status_code != 200:
    print(f"Error: {response.status_code}, Response: {response.text}")
    response.raise_for_status()

transcript_id = response.json()["id"]
polling_endpoint = f"{endpoint}/{transcript_id}"

while True:
    transcript = requests.get(polling_endpoint, headers=headers).json()
    if transcript["status"] == "completed":
        # for result in transcript['content_safety_labels']['results']:
        #     print(result['text'])
        #     print(f"Timestamp: {result['timestamp']['start']} - {result['timestamp']['end']}")
        #     # Get category, confidence, and severity.
        #     for label in result['labels']:
        #         print(f"{label['label']} - {label['confidence']} - {label['severity']}")  # content safety category
        pprint.pprint(transcript)
        print(transcript["text"])
        break
    elif transcript["status"] == "error":
        raise RuntimeError(f"Transcription failed: {transcript['error']}")
        break
    else:
        print("Waiting for 3 seconds")
        time.sleep(3)
    
#profane_words = set(np.genfromtxt("words_to_block/Terms-to-Block.csv", delimiter=',', dtype=str, encoding=None))
df = pd.read_csv("words_to_block/profanity_en.csv", usecols=[0], header=None)
profane_words = set(df[0].dropna().str.strip())

processed_words = {helpers.process_text(word) for word in profane_words}


time_stamps = []
for i in range(len(transcript['words'])):
    text = helpers.process_text(transcript['words'][i]['text'])
    if text in processed_words:
        time_stamps.append([transcript['words'][i]['start'], transcript['words'][i]['end']])

print(time_stamps)

helpers.process_audio(audio, time_stamps)

rm_file = "data/audio/audio_to_process.wav"
if os.path.exists(rm_file):
    os.remove(rm_file)
else:
    print("File not found")