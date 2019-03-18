import librosa
import math
import numpy as np
from madmom.features.onsets import OnsetPeakPickingProcessor, RNNOnsetProcessor

#nothing more precise than a 32nd note in our step charts
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


def onsets_to_amplitude(onsets, path):
    """
    :param onsets: list of times where onsets occur
    :param path: path for audio file
    :return: list of signal amplitudes at times in onsets list
    """
    y, sr = librosa.load(path, sr=None)
    samples_in_time = librosa.core.samples_to_time(y, sr=sr)
    wanted_indexs = np.searchsorted(samples_in_time, onsets)
    amp_array = y[wanted_indexs]
    return amp_array


def precision_times(path, bpm):
    """
    Finds list of times where 16th notes can occur in the track
    :param path: Path to audio file
    :param bpm: BPM of the track
    :return: List of times (in seconds) where note of the predetermined precision may occur
    """
    y, sr = librosa.load(path, sr=None)
    duration = librosa.core.get_duration(y=y, sr=sr)

    # multiply by 4 because our precision is 16th notes which is 4x more precise than quarter
    per_second = bpm * (1 / 60.0) * (precision/4)
    increment = 1 / per_second
    num_precision_beats = int(math.floor(duration * per_second))
    return [i * increment for i in range(num_precision_beats)]


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

def correct_note_density(onsets,path,bpm,rating):
    """
    Corrects regions of song with too high a note density for too long
    :param onsets: List of times (in seconds) corresponding to onsets
    :return: onsets: subset of input onsets list
    """
    if rating == "easy":
        max_note_density = 1
    elif rating == "medium":
        max_note_density = 3
    else: #hard
        max_note_density = 4.75
    new_onsets = np.array([])
    #make sure onsets is a numpy array
    onsets = np.array(onsets)
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

def onsets_to_notes(onsets, bpm, path,rating):
    """
    Finds 32nd notes to place arrows at
    :param onsets: List of times (in seconds) corresponding to onsets
    :param bpm: BPM of the track
    :param path: Path to audio file
    :return: List of 32nd note indices to place arrows at, list of onsets that don't exceed a given note density
    """
    possible_times = precision_times(path, bpm)
    onsets = correct_note_density(onsets,path,bpm,rating)
    notes = [np.searchsorted(possible_times, onset) for onset in onsets]

    return notes, onsets
