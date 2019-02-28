import librosa
import pydub
import madmom
import numpy as np
from madmom.features.beats import BeatTrackingProcessor
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor
from madmom.features.onsets import OnsetPeakPickingProcessor, RNNOnsetProcessor
from pydub import AudioSegment


def get_bpm(path):
    proc = TempoEstimationProcessor(fps=100)
    act = RNNBeatProcessor()(path)  # returns activation function
    tempos = proc(act)  # pass activations to processor to get tempo information
    tempo = round(tempos[0, 0], 4)
    if tempo < 100:  # slow songs are less fun
        tempo *= 2
    return tempo


def get_beats(path):
    proc = BeatTrackingProcessor(fps=100)
    act = RNNBeatProcessor()(path)
    beat_probabilities = proc(act)
    return beat_probabilities