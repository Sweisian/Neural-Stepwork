import librosa
import math
import numpy as np
import random
from madmom.features.onsets import OnsetPeakPickingProcessor, RNNOnsetProcessor

precision = 32

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
    sixteenths_per_second = bpm * (1 / 60.0) * (precision/4)
    increment = 1 / sixteenths_per_second
    num_16_beats = math.floor(duration * sixteenths_per_second)

    return [i * increment for i in range(num_16_beats)]


def onsets_to_amplitudes(onsets, path):
    """
    :param onsets: List of times (in seconds) corresponding to onsets
    :param path: location to audio file
    :return: List of amplitudes corresponding to times where onsets occur
    """
    y, sr = librosa.load(path, sr=None)
    duration = librosa.core.get_duration(y=y, sr=sr)
    samples_in_time = librosa.core.samples_to_time(np.arange(y.size), sr=sr)
    wanted_indices = np.searchsorted(samples_in_time, onsets)
    # care about signal energy, so take abs value
    amp_array = np.abs(y[wanted_indices])
    amp_dict = {key: value for key, value in zip(onsets, amp_array)}
    return amp_dict, amp_array, duration

def correct_note_density(onsets,path,bpm):
    """
    Corrects regions of song with too high a note density for too long
    :param onsets: List of times (in seconds) corresponding to onsets
    :return: onsets: subset of input onsets list
    """
    new_onsets = np.array([])
    #make sure onsets is a numpy array
    onsets = np.array(onsets)
    # 5 arrows per second
    max_note_density = 4.75
    amp_dict, amplitudes, duration = onsets_to_amplitudes(onsets,path)
    window_size = duration / (bpm * duration / 240) #length of a measure
    t = 0
    while t < (duration - window_size):
        window = onsets[(onsets> t) & (onsets < (t+window_size))]
        window_amp = []
        new_window = np.array([])
        for w in window:
            window_amp.append(amp_dict[w])
        note_density = len(window) / window_size
        if note_density > max_note_density:
            p = np.percentile(window_amp, (note_density - max_note_density) / note_density * 100)
            for w in window:
                if amp_dict[w] > p:
                    new_window = np.append(new_window,w)
        t+=window_size
        if len(new_window) == 0:
            new_window = window
        new_onsets = np.append(new_onsets,new_window)
    return new_onsets

def onsets_to_notes(onsets, bpm, path):
    """
    Finds 16th notes to place arrows at
    :param onsets: List of times (in seconds) corresponding to onsets
    :param bpm: BPM of the track
    :param path: Path to audio file
    :return: List of 16th note indices to place arrows at
    """
    possible_times = times16ths(path, bpm)
    onsets = correct_note_density(onsets,path,bpm)
    notes = [np.searchsorted(possible_times, onset) for onset in onsets]
    return notes

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
    last = 2 #randomly generate arrows that are different from the last arrow
    for i in range(len(notes) - 1):
        r = randomArrow(last)
        last = r
        chart[int(notes[i])][r] = 1
    measures = ""
    i = 0
    while i < (len(chart) - precision):
        measure = ""
        for j in range(precision):
            measure += str(chart[i + j])[1:-1] + "\n"
        measures += measure + "\n,\n"
        i += precision
    return measures[:-2]

def randomArrow(last):
    r = random.randint(0, 3)
    while (r == last):
        r = random.randint(0, 3)
    return r
