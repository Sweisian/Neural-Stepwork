import librosa
import math
import numpy as np
import random
from madmom.features.onsets import OnsetPeakPickingProcessor, RNNOnsetProcessor
from bisect import bisect_left


def get_onsets(path):
    """
    Finds onsets of audio file
    :param path: Path to audio file
    :return: List of times (in seconds) corresponding to onsets
    """
    processor = OnsetPeakPickingProcessor(fps=100)
    act = RNNOnsetProcessor()(path)
    onsets = processor(act)
    return onsets


def times16ths(path, bpm):
    """
    Finds list of times where 16th notes can occur in the track
    :param path: Path to audio file
    :param bpm: BPM of the track
    :return: List of times (in seconds) where 16th note occur
    """
    y, sr = librosa.load(path, sr=None)
    duration = librosa.core.get_duration(y=y, sr=sr)

    # multiply by 4 because our precision is 16th notes which is 4x more precise than quarter
    sixteenths_per_second = bpm * (1 / 60.0) * 4
    increment = 1 / sixteenths_per_second
    num_16_beats = math.floor(duration * sixteenths_per_second)

    return [i * increment for i in range(num_16_beats)]


def onsets_to_notes(onsets, bpm, path):
    """
    Finds 16th notes to place arrows at
    :param onsets: List of times (in seconds) corresponding to onsets
    :param bpm: BPM of the track
    :param path: Path to audio file
    :return: List of 16th note indices to place arrows at
    """
    possible_times = times16ths(path, bpm)
    return [np.searchsorted(possible_times, onset) for onset in onsets ]


def notes_to_measures(notes, bpm, path):
    """
    Creates string of notes to write to sm file
    :param notes: List of 16th note indices to place arrows at
    :param bpm: BPM of the track
    :param path: Path to audio file
    :return: String representation of a matrix
    """
    num_16ths = len(times16ths(path, bpm))
    chart = np.zeros((num_16ths + 1, 4))
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
