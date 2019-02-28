import librosa
import pydub
import math
import madmom
import numpy as np
import random
from madmom.features.beats import BeatTrackingProcessor
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor
from madmom.features.onsets import OnsetPeakPickingProcessor, RNNOnsetProcessor
import find_bpm
from pydub import AudioSegment
import find_bpm


def get_onsets(path):
    proc = OnsetPeakPickingProcessor(fps=100)
    act = RNNOnsetProcessor()(path)
    onsets = proc(act)
    # print(16,onsets)
    # print(16,onsets)break
    return onsets

def swei_onsets_to_measures(onset_time_array, bpm, path):
    y, sr = librosa.load(path, sr=None)
    song_len = len(y) / sr
    num_beats = math.floor(bpm * (1 / 60) * song_len)
    num_measures = math.floor(num_beats / 4)
    num_16_notes = num_measures * 16
    notes_16 = np.zeros(num_16_notes)

    time_gap_16 = song_len/notes_16.size

    for idx, ele in enumerate(notes_16):
        notes_16[idx] = idx * time_gap_16

    #print(notes_16.size)
    #print(notes_16.shape[0])
    #print(len(notes_16))
    measure_array = np.zeros((notes_16.size, 4))
    #print(measure_array)
    for ele in onset_time_array:
        note_idx = np.searchsorted(notes_16, ele)
        #This is where step selection will eventually go in some format
        if note_idx < measure_array.size:
            measure_array[note_idx, 0] = 1

    measures = ""
    for row in measure_array:
        measures += str(row)[1:-1] + "\n,\n"
    return measures[:-2]


path = "train/60_TheFatRat_Unity.wav"
onset_time_array = get_onsets(path)
bpm = find_bpm.get_bpm(path)
measures = swei_onsets_to_measures(onset_time_array, bpm, path)
print(measures)



def onsets_to_notes(onsets, bpm):
    notes = np.zeros(np.shape(onsets))
    for i in range(len(notes)):
        notes[i] = int(bpm * (1 / 60.0) * onsets[i])
    return notes


def notes_to_measures(notes, bpm, path):
    y, sr = librosa.load(path, sr=None)
    song_len = len(y) / sr
    num_beats = math.floor(bpm * (1 / 60) * song_len)
    chart = np.zeros((num_beats, 4))
    for i in range(num_beats):
        for j in notes[i:]:
            j = int(j)
            r = random.randint(0, 3)
            if j == i:
                chart[i][r] = 1
            else:
                chart[i][r] = 0
            if j > i:
                break
    measures = ""
    for row in chart:
        measures += str(row)[1:-1] + "\n,\n"
    return measures[:-2]
