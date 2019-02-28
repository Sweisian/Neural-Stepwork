# import modules
import librosa
import numpy as np
import scipy.io.wavfile
import madmom
from onset_detection import get_onsets



def wavwrite(filepath, data, sr, norm=True, dtype='int16', ):
    '''
    Write wave file using scipy.io.wavefile.write, converting from a float (-1.0 : 1.0) numpy array to an integer array

    Parameters
    ----------
    filepath : str
        The path of the output .wav file
    data : np.array
        The float-type audio array
    sr : int
        The sampling rate
    norm : bool
        If True, normalize the audio to -1.0 to 1.0 before converting integer
    dtype : str
        The output type. Typically leave this at the default of 'int16'.
    '''
    if norm:
        data /= np.max(np.abs(data))
    data = data * np.iinfo(dtype).max
    data = data.astype(dtype)
    scipy.io.wavfile.write(filepath, sr, data)

# read audio file


####################### LIBROSA #############################
# # approach 1 - onset detection and dynamic programming
# tempo, beat_times = librosa.beat.beat_track(x, sr=sr, units='time')
# clicks = librosa.clicks(beat_times, sr=sr, length=len(x))


###################### MADMOM ###############################
# approach 2 - dbn tracker

def add_beat_clicks():
    x, sr = librosa.load('train/TheFatRat_Unity.wav', duration=60)
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()("train/TheFatRat_Unity.wav")

    beat_times = proc(act)

    clicks = librosa.clicks(beat_times, sr=sr, length=len(x))
    song_with_clicks = x + clicks
    return song_with_clicks


def add_onset_clicks():
    x, sr = librosa.load('train/TheFatRat_Unity.wav', duration=60)

    onset_times = get_onsets('train/TheFatRat_Unity.wav')

    clicks = librosa.clicks(onset_times, sr=sr, length=len(x))
    song_with_clicks = x + clicks

    wavwrite('output/TheFatRat_Unity_Onset_clicks.wav', song_with_clicks, sr, norm=True, dtype='int16')

add_onset_clicks()