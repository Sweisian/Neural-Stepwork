import librosa
import math
import numpy as np
import random
from madmom.features.onsets import OnsetPeakPickingProcessor, RNNOnsetProcessor

# returns list of onsets in seconds
def get_onsets(path):
    proc = OnsetPeakPickingProcessor(fps=100)
    act = RNNOnsetProcessor()(path)
    onsets = proc(act)
    return onsets


# returns list of times that all 16th notes occur at in the song
def times16ths(path, bpm):
    times = []
    y, sr = librosa.load(path, sr=None)
    duration = librosa.core.get_duration(y=y, sr=sr)
    # find number of 16th notes per second
    # multiply by 4 because our precision is 16th notes which is 4x more precise than quarter
    bps = (bpm * (1 / 60.0) * 4)
    increment = 1 / bps
    num16Beats = int(math.floor(duration * bps))
    for i in range(num16Beats):
        times.append(i * increment)
    return times


# convert onsets (in seconds) to notes (the nth 16th note in the song)
def onsets_to_notes(onsets, bpm, path):
    notes = np.zeros(len(onsets))
    possibleTimes = times16ths(path, bpm)
    for i in range(len(notes)):
        notes[i] = np.searchsorted(possibleTimes, onsets[i])
    return notes

# given a series of notes, create a series of measures as a string (1 where there is a note, 0 where there isnt)
# for now, we randomly decide which arrow type (up, down, etc.) to use
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
