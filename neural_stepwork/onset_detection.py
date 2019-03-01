import librosa
import math
import numpy as np
import random
from madmom.features.onsets import OnsetPeakPickingProcessor, RNNOnsetProcessor


def get_onsets(path):
    proc = OnsetPeakPickingProcessor(fps=100)
    act = RNNOnsetProcessor()(path)
    onsets = proc(act)
    # print(16,onsets)
    # print(16,onsets)break
    return onsets


def times16ths(path, bpm):
    times = []
    y, sr = librosa.load(path, sr=None)
    duration = librosa.core.get_duration(y=y, sr=sr)
    print("duraction=", duration)
    bps = (
        bpm * (1 / 60.0) * 4
    )  # multiply by 4 because our precision is 16th notes which is 4x more precise than quarter
    increment = 1 / bps
    # print("incr=",increment)
    num16Beats = math.floor(duration * bps)
    # print("num 16th beats",num16Beats)
    for i in range(num16Beats):
        times.append(i * increment)
    return times


def onsets_to_notes(onsets, bpm, path):
    notes = np.zeros(len(onsets))
    possibleTimes = times16ths(path, bpm)
    # print("onsets = ",onsets[:100])
    # print("possible times = ",possibleTimes[:20])
    for i in range(len(notes)):
        # notes[i] = int(bpm * (1 / 60.0) * onsets[i])
        notes[i] = np.searchsorted(possibleTimes, onsets[i])
    # print(28,"num onsets=",len(onsets),"num notes=",len(notes),"unique notes=",len(list(set(notes))))
    # print(notes)
    return notes


# path = "songs/Nekozilla.wav"
# onsets_to_notes(get_onsets(path),128,path)


def notes_to_measures(notes, bpm, path):
    num_beats = len(times16ths(path, bpm))
    chart = np.zeros((num_beats + 1, 4))
    print("chart size=", np.shape(chart))
    for i in range(len(notes) - 1):
        r = random.randint(0, 3)
        # print(notes[i])
        chart[int(notes[i])][r] = 1
    measures = ""
    i = 0
    # print("len chart=",len(chart))
    while i < (len(chart) - 16):
        measure = ""
        for j in range(16):
            # print("i=",i,"j=",j)
            measure += str(chart[i + j])[1:-1] + "\n"
        measures += measure + "\n,\n"
        i += 16
    return measures[:-2]


# print(notes_to_measures([1,2,3,4,5],123,path))
