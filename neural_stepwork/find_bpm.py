from madmom.features.beats import BeatTrackingProcessor
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor



def get_bpm(path):
    proc = TempoEstimationProcessor(fps=100)
    # returns activation function
    act = RNNBeatProcessor()(path)
    # pass activations to processor to get tempo information
    tempos = proc(act)
    # at index [0][0] is the min bpm found in the song
    tempo = round(tempos[0, 0], 4)
    # slow songs are less fun
    if tempo < 110:
        tempo *= 2
    return tempo

