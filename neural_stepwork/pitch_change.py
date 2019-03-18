import numpy as np
from scipy.signal import stft
import librosa
from .onset_detection import get_onsets, onsets_to_notes
from .find_bpm import get_bpm


def pitch_change(path,onsets,rating):
    """
    :param path: path to audio file
    :param onsets: list of times where onsets occur in seconds
    :param rating: difficulty rating of chart (easy, medium, hard)
    :return: list of changes in frequency between onset times
    """
    if len(onsets) == 0:
        onsets = get_onsets(path)
    notes, onsets = onsets_to_notes(onsets, get_bpm(path), path,rating)
    x, sr = librosa.load(path, sr=None)
    f,t, zxx = stft(x=x,fs=sr)
    freq = []
    indices = np.searchsorted(t,onsets)
    for i in indices:
        freq.append(max(np.abs(zxx[:,i])))
    change = []
    for i in range(1,len(freq)):
        prev = freq[i-1]
        curr = freq[i]
        change.append(curr-prev)
    return change
