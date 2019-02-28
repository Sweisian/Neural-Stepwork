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
from pydub import AudioSegment
import bpmFinder

def getOnsets(path):
    proc = OnsetPeakPickingProcessor(fps=100)
    act = RNNOnsetProcessor()(path)
    onsets = proc(act)
    #print(16,onsets)
    #print(16,onsets)break
    return onsets

def onsetsToNotes(onsets,bpm):
    notes = np.zeros(np.shape(onsets))
    for i in range(len(notes)):
        notes[i] = int(bpm * (1/60.0) * onsets[i])
    return notes

def notesToMeasures(notes,BPM,path):
    y, sr = librosa.load(path, sr=None)
    songLength = len(y)/sr
    numBeats = math.floor(BPM * (1/60) * songLength)
    chart = np.zeros((numBeats,4))
    for i in range(numBeats):
        for j in notes[i:]:
            j = int(j)
            r = random.randint(0,3)
            if j == i:
                chart[i][r] = 1
            else:
                chart[i][r] = 0
            if j > i:
                break
    measures = ""
    for row in chart:
        measures+= (str(row)[1:-1] + "\n,\n")
    #print(measures)
    return measures[:-2]

