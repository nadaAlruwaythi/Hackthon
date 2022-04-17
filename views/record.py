import fastapi
from fastapi_chameleon import template
from starlette.requests import Request

from viewmodels.home.indexviewmodel import IndexViewModel
from viewmodels.shared.viewmodel import ViewModelBase
from fastapi_chameleon import template
from fastapi import FastAPI ,File, UploadFile ,Request
from starlette.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, Form, UploadFile
import os
import pickle
import warnings
import requests
import json
import speech_recognition as sr
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import *
from matplotlib import pyplot as plt
import torch
import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio
import shutil
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer ,Wav2Vec2Processor
from fastapi.responses import ORJSONResponse
TEMPLATES = Jinja2Templates(directory=( "templates"))


router = fastapi.APIRouter()

#
# @router.get('/record')
# @template()
# def index(request : Request):
#     print('Hi')
#     return {}
#
#
#
#
# @router.post("/record", response_class=ORJSONResponse)
# async def create_upload_file(request: Request,file: UploadFile = File(...)):
#     with open("1.wav", "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     processor = Wav2Vec2Processor.from_pretrained("EngNada/wav2vec2-large-xlsr-53-demo1")
#     model = Wav2Vec2ForCTC.from_pretrained("EngNada/wav2vec2-large-xlsr-53-demo1", num_labels=1)
#
#     wavFile = '1.wav'
#
#     """## start of piplien"""
#
#     # Commented out IPython magic to ensure Python compatibility.
#     segLen, frameRate, numMix = 3, 50, 330
#
#     def VoiceActivityDetection(wavData, frameRate):
#         # uses the librosa library to compute short-term energy
#         ste = librosa.feature.rms(wavData, hop_length=int(16000 / frameRate)).T
#         thresh = 0.1 * (np.percentile(ste, 97.5) + 9 * np.percentile(ste,
#                                                                      2.5))  # Trim 5% off and set threshold as 0.1x of the ste range
#         return (ste > thresh).astype('bool')
#
#     wavData, _ = librosa.load(wavFile, sr=16000)
#     vad = VoiceActivityDetection(wavData, frameRate)
#
#     mfcc = librosa.feature.mfcc(wavData, sr=16000, n_mfcc=20, hop_length=int(16000 / frameRate)).T
#     vad = np.reshape(vad, (len(vad),))
#     if mfcc.shape[0] > vad.shape[0]:
#         vad = np.hstack((vad, np.zeros(mfcc.shape[0] - vad.shape[0]).astype('bool'))).astype('bool')
#     elif mfcc.shape[0] < vad.shape[0]:
#         vad = vad[:mfcc.shape[0]]
#     mfcc = mfcc[vad, :];
#     n_components = np.arange(1, 21)
#     models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(mfcc)
#               for n in n_components]
#
#     def trainGMM(wavFile, frameRate, segLen, vad, numMix):
#         wavData, _ = librosa.load(wavFile, sr=16000)
#         mfcc = librosa.feature.mfcc(wavData, sr=16000, n_mfcc=20, hop_length=int(16000 / frameRate)).T
#         vad = np.reshape(vad, (len(vad),))
#         if mfcc.shape[0] > vad.shape[0]:
#             vad = np.hstack((vad, np.zeros(mfcc.shape[0] - vad.shape[0]).astype('bool'))).astype('bool')
#         elif mfcc.shape[0] < vad.shape[0]:
#             vad = vad[:mfcc.shape[0]]
#         mfcc = mfcc[vad, :];
#         print("Training GMM..")
#         GMM = GaussianMixture(n_components=numMix).fit(mfcc)
#         var_floor = 1e-5
#         segLikes = []
#         segSize = frameRate * segLen
#         for segI in range(int(np.ceil(float(mfcc.shape[0]) / (frameRate * segLen)))):
#             startI = segI * segSize
#             endI = (segI + 1) * segSize
#             if endI > mfcc.shape[0]:
#                 endI = mfcc.shape[0] - 1
#             if endI == startI:  # Reached the end of file
#                 break
#             seg = mfcc[startI:endI, :]
#             compLikes = np.sum(GMM.predict_proba(seg), 0)
#             segLikes.append(compLikes / seg.shape[0])
#         print("Training Done")
#
#         return np.asarray(segLikes)
#
#     clusterset = trainGMM(wavFile, frameRate, segLen, vad, numMix)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(clusterset)
#     X_normalized = normalize(X_scaled)
#     cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
#     clust = cluster.fit_predict(X_normalized)
#
#     def SegmentFrame(clust, segLen, frameRate, numFrames):
#         frameClust = np.zeros(numFrames)
#         for clustI in range(len(clust) - 1):
#             frameClust[clustI * segLen * frameRate:(clustI + 1) * segLen * frameRate] = clust[clustI] * np.ones(
#                 segLen * frameRate)
#         frameClust[(clustI + 1) * segLen * frameRate:] = clust[clustI + 1] * np.ones(
#             numFrames - (clustI + 1) * segLen * frameRate)
#         return frameClust
#
#     frameClust = SegmentFrame(clust, segLen, frameRate, mfcc.shape[0])
#
#     frameClust = SegmentFrame(clust, segLen, frameRate, mfcc.shape[0])
#
#     def speakerdiarisationdf(hyp, frameRate, wavFile):
#         audioname = []
#         starttime = []
#         endtime = []
#         speakerlabel = []
#
#         spkrChangePoints = np.where(hyp[:-1] != hyp[1:])[0]
#         if spkrChangePoints[0] != 0 and hyp[0] != -1:
#             spkrChangePoints = np.concatenate(([0], spkrChangePoints))
#         spkrLabels = []
#         for spkrHomoSegI in range(len(spkrChangePoints)):
#             spkrLabels.append(hyp[spkrChangePoints[spkrHomoSegI] + 1])
#         for spkrI, spkr in enumerate(spkrLabels[:-1]):
#             if spkr != -1:
#                 audioname.append(wavFile.split('/')[-1].split('.')[0] + ".wav")
#                 starttime.append((spkrChangePoints[spkrI] + 1) / float(frameRate))
#                 endtime.append((spkrChangePoints[spkrI + 1] - spkrChangePoints[spkrI]) / float(frameRate))
#                 speakerlabel.append("المتحدث" + str(int(spkr)))
#         if spkrLabels[-1] != -1:
#             audioname.append(wavFile.split('/')[-1].split('.')[0] + ".wav")
#             starttime.append(spkrChangePoints[-1] / float(frameRate))
#             endtime.append((len(hyp) - spkrChangePoints[-1]) / float(frameRate))
#             speakerlabel.append("المتحدث " + str(int(spkrLabels[-1])))
#         #
#         speakerdf = pd.DataFrame(
#             {"Audio": audioname, "starttime": starttime, "endtime": endtime, "speakerlabel": speakerlabel})
#
#         spdatafinal = pd.DataFrame(columns=['Audio', 'SpeakerLabel', 'StartTime', 'EndTime'])
#         i = 0
#         k = 0
#         j = 0
#         spfind = ""
#         stime = ""
#         etime = ""
#         for row in speakerdf.itertuples():
#             if (i == 0):
#                 spfind = row.speakerlabel
#                 stime = row.starttime
#             else:
#                 if (spfind == row.speakerlabel):
#                     etime = row.starttime
#                 else:
#                     spdatafinal.loc[k] = [wavFile.split('/')[-1].split('.')[0] + ".wav", spfind, stime, row.starttime]
#                     k = k + 1
#                     spfind = row.speakerlabel
#                     stime = row.starttime
#             i = i + 1
#         spdatafinal.loc[k] = [wavFile.split('/')[-1].split('.')[0] + ".wav", spfind, stime, etime]
#         return spdatafinal
#
#     pass1hyp = -1 * np.ones(len(vad))
#     pass1hyp[vad] = frameClust
#     spkdf = speakerdiarisationdf(pass1hyp, frameRate, wavFile)
#     spkdf["TimeSeconds"] = spkdf.EndTime - spkdf.StartTime
#
#     spkdf
#
#     if not os.path.exists("Vt"):
#         os.makedirs("Vt")
#     list1 = []
#
#     def map_spkr_trans(spkdf):
#         for i in range(len(spkdf)):
#             if (spkdf.iloc[i]['TimeSeconds'] > 0):
#                 t1 = spkdf.iloc[i]['StartTime'] * 1000  # Works in milliseconds
#                 t2 = spkdf.iloc[i]['EndTime'] * 1000
#                 newAudio = AudioSegment.from_wav(wavFile)
#                 newAudio = newAudio[t1:t2]
#                 newAudio.export('Vt/' + str(i) + '.wav', format="wav")
#                 samples, sample_rate = librosa.load('Vt/' + str(i) + '.wav', sr=16000)
#                 input_values = processor(samples, return_tensors="pt").input_values
#                 logits = model(input_values).logits
#                 predicted_ids = torch.argmax(logits, dim=-1)
#                 transcription = processor.batch_decode(predicted_ids)[0]
#                 print(spkdf.iloc[i]['SpeakerLabel'] + ' : ' + transcription)
#                 list1.append(spkdf.iloc[i]['SpeakerLabel'] + ' : ' + transcription)
#
#
#
#     map_spkr_trans(spkdf)
#     print(list1)
#     return TEMPLATES.TemplateResponse(
#         "res.html", {"request": request, "lists": list1},
#
#     )


import numpy as np
import pandas as pd
from IPython.display import Audio
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler
import sys
from scipy.io import wavfile
import math
from fastapi.templating import Jinja2Templates

import fastapi
from fastapi_chameleon import template
from starlette.requests import Request

from viewmodels.home.indexviewmodel import IndexViewModel
from viewmodels.shared.viewmodel import ViewModelBase
from fastapi_chameleon import template
from fastapi import FastAPI ,File, UploadFile ,Request
from starlette.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, Form, UploadFile
import os

@router.post("/record", response_class=ORJSONResponse)



AUDIO_FILE = './Vt/1.wav'
y, sr = librosa.load(AUDIO_FILE)
Audio(data=y, rate=sr)
totDuration = librosa.get_duration(filename=AUDIO_FILE)
print("Total Duration(s): ", totDuration)
# Create chunks/windows for feeding into the model
division_per_second = 1
chunk_time = 1.0 / division_per_second
chunk_size = sr // division_per_second
print('Samples in one chunk (chunk size)', chunk_size)
print('Duration of one chunk', chunk_time)
print('Number of chunks: ', y.shape[0] // chunk_size)

remainder_chunks = y.shape[0] % chunk_size
num_of_chunks = 1
if (remainder_chunks > 0):
    num_of_chunks = y[:-remainder_chunks].shape[0] / chunk_size
    Y = np.split(y[:-remainder_chunks], num_of_chunks)
else:
    num_of_chunks = y.shape[0] / chunk_size
    Y = np.split(y, num_of_chunks)
print(chunk_size)
len(Y)
print(Y[0])

# Extract feature: Mel-frequency Cepstral Coefficients
feature_mfcc = np.array([librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=40) for chunk in Y])
feature_mfcc.shape
feature_mfcc_mean = np.mean(feature_mfcc, axis=2)
print(feature_mfcc_mean.shape)
# Extract feature: Spectral flatness
feature_spectral_flatness = np.array([librosa.feature.spectral_flatness(y=y) for chunk in Y])
feature_spectral_flatness.shape
feature_spectral_flatness_mean = np.mean(feature_spectral_flatness, axis=2)
print(feature_spectral_flatness_mean.shape)
# Extract feature: Spectral flux
feature_specflux = np.array([librosa.onset.onset_strength(y=y, sr=sr) for chunk in Y])
feature_specflux.shape
feature_specflux_mean = np.mean(feature_specflux, axis=1).reshape(-1, 1)
print(feature_specflux_mean.shape)
# Extract feature: Pitch
feature_pitches = np.array([librosa.piptrack(y=y, sr=sr)[0] for chunk in Y])
feature_pitches.shape
feature_pitches_mean = np.mean(feature_pitches, axis=2)
print(feature_pitches_mean.shape)
# Create final feature space for feeding into the model
X = np.hstack((
    feature_mfcc_mean,
    feature_spectral_flatness_mean,
    feature_specflux_mean,
    feature_pitches_mean
))
X.shape
# Normalize the input
scaler = StandardScaler()
X = scaler.fit_transform(X)
X.shape
from sklearn.mixture import GaussianMixture

nclusters = 6
gmm = GaussianMixture(n_components=nclusters)
gmm.fit(X)

op = gmm.predict(X)
time_speaker = {}
for i in range(len(op)):
    time_speaker[i + 1] = str(op[i])

print(time_speaker)


# Segmentation Algorithm

def getCount(time_speaker, cluster, wstart, wend):
    count = 0
    lastSeenAt = None
    for i in range(wstart, wend + 1):
        if (time_speaker[i] == cluster):
            count += 1
            lastSeenAt = i
    return (count, lastSeenAt)


def getSuccessor(time_speaker, currentCluster, wstart, params):
    totalDuration = len(time_speaker)
    lookahead = int(params['lookaheadTime'] / chunk_time)
    i = wstart
    successorCount = 0
    while (i <= totalDuration - lookahead):
        j = i + lookahead
        successor = time_speaker[i]
        if (successor == currentCluster):
            return (i, successor)
        successorCount = getCount(time_speaker, successor, i, j)
        if (successorCount[0] > int(params['epsilon'] / chunk_time)):
            return (i, successor)
        i += 1
    i = min(i + lookahead, totalDuration)
    return (i, time_speaker[i])


def getBreakPoint(time_speaker, cluster, wstart, params):
    totalDuration = len(time_speaker)
    i = wstart
    end = None
    while (time_speaker[i] == cluster):
        i += 1
    breaker = time_speaker[i]
    j = min(i + int(params['lookaheadTime'] / chunk_time), totalDuration)
    breakerCount = getCount(time_speaker, breaker, i, j)
    clusterCount = getCount(time_speaker, cluster, i, j)
    if (breakerCount[0] >= int(params['epsilon'] / chunk_time)):
        end = i
    else:
        i += 1
        successor = getSuccessor(time_speaker, cluster, i, params)
        if (successor[1] == cluster):
            i = successor[0]
            end = getBreakPoint(time_speaker, cluster, i, params)
        else:
            end = successor[0]
    return end


def segment(time_speaker, params):
    segments = {}
    totalDuration = len(time_speaker)
    w = 1
    recorded = {}
    while (w <= totalDuration):
        cluster = time_speaker[w]
        print('Current cluster: ', cluster)
        start = None
        end = None
        if (recorded.get(cluster) == None):
            i = w
            j = min(w + int(params['lookaheadTime'] / chunk_time), totalDuration)
            count = getCount(time_speaker, cluster, i, j)
            print("Count in lookahead window: " + 'cluster: ' + cluster, count[0])
            if (count[0] > int(params['epsilon'] / chunk_time)):
                start = i
                end = getBreakPoint(time_speaker, cluster, start, params)
                w = end - 1
                print('End of ' + cluster + ' is : ', end)
                segments[cluster] = (start, end)
                recorded[cluster] = True
                continue
        w += 1
    return segments


params = {
    'lookaheadTime': 7,
    'epsilon': 4
}
segments = segment(time_speaker, params)
segments

speakers = {}
rate, data = wavfile.read('./Vt/1.wav')
i = 1
for k in segments:
    start = math.ceil(rate * segments[k][0] * chunk_time)
    end = math.ceil(rate * segments[k][1] * chunk_time)
    speakers[i] = data[start:end + 1]
    i += 1

speakers
for i in speakers:
    wavfile.write('Vt/' + str(i) + '.wav', rate, speakers[i])

len(speakers)

import speech_recognition as sr
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import pandas as pd

# speech recognizer
recognizer = sr.Recognizer()
directoryPath = "./Vt/"


# split audio file into chunks and convert audio to text
def get_audio_paths(path):

    audio = AudioSegment.from_wav(path)
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(audio, min_silence_len=1000, silence_thresh=audio.dBFS - 9, keep_silence=800)
    folder_name = directoryPath + "/chunks"
    # store chunks in the following directory (create if not exists)
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    final_text = ""
    # for each chunk
    for i, chunk in enumerate(chunks, start=1):

        filename = os.path.join(folder_name, f"chunk{i}.wav")
        chunk.export(filename, format="wav")
        # recognize the audio
        with sr.AudioFile(filename) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio_listened = recognizer.record(source)
            # convert audio to text
            try:
                text = recognizer.recognize_google(audio_listened, language='ar')
                print(text)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                final_text += text.lower() + "."
    return final_text


paths = []
for i, file in enumerate(os.listdir(directoryPath)):
    if file.endswith(".wav"):
        print(os.path.basename(file))
        paths.append(os.path.basename(file))

list1 = []
for i, path in enumerate(paths):
    list1.append(get_audio_paths(directoryPath + path))

dataframe = pd.DataFrame(list1, columns=['Text'])
dataframe.to_csv("sampt.csv")

# for text in list1:
#     print(text)

return TEMPLATES.TemplateResponse(
        "res.html", {"request": request, "lists": list1},

    )

