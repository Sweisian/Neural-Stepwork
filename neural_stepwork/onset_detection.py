import librosa
import math
import numpy as np
import random
from madmom.features.onsets import OnsetPeakPickingProcessor, RNNOnsetProcessor


def get_onsets(path):
    processor = OnsetPeakPickingProcessor(fps=100)
    act = RNNOnsetProcessor()(path)
    onsets = processor(act)
    return onsets


def times16ths(path, bpm):
    times = []
    y, sr = librosa.load(path, sr=None)
    duration = librosa.core.get_duration(y=y, sr=sr)
    print("duration =", duration)
    # multiply by 4 because our precision is 16th notes which is 4x more precise than quarter
    bps = bpm * (1 / 60.0) * 4
    increment = 1 / bps
    num_16_beats = math.floor(duration * bps)

    for i in range(num_16_beats):
        times.append(i * increment)

    return times


def onsets_to_notes(onsets, bpm, path):
    notes = np.zeros(len(onsets))
    possible_times = times16ths(path, bpm)
    for i in range(len(notes)):
        notes[i] = np.searchsorted(possible_times, onsets[i])
    return notes


def notes_to_measures(notes, bpm, path):
    num_beats = len(times16ths(path, bpm))
    chart = np.zeros((num_beats + 1, 4))
    print("chart size=", np.shape(chart))
    for i in range(len(notes) - 1):
        r = random.randint(0, 3)
        chart[int(notes[i])][r] = 1
    measures = ""
    i = 0
    while i < (len(chart) - 16):
        measure = ""
        for j in range(16):
            measure += str(chart[i + j])[1:-1] + "\n"
        measures += measure + "\n,\n"
        i += 16
    return measures[:-2]
