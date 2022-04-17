import numpy as np
import pandas as pd
from IPython.display import Audio
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler
import sys
from scipy.io import wavfile
import math

AUDIO_FILE = 'Vt/.wav'
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
rate, data = wavfile.read('Vt/1.wav')
i = 1
for k in segments:
    start = math.ceil(rate * segments[k][0] * chunk_time)
    end = math.ceil(rate * segments[k][1] * chunk_time)
    speakers[i] = data[start:end + 1]
    i += 1

speakers
for i in speakers:
    wavfile.write('/content/data' + str(i) + '.wav', rate, speakers[i])

len(speakers)

