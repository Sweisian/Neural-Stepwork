#import madmom
import librosa
import numpy as np

#off for BreakDown
#correct for Nekozilla

y,sr = librosa.load("songs/BreakDown.wav",sr=None)
tempo, beats = librosa.beat.beat_track(y,sr)
print(tempo)

