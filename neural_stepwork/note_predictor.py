import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.models import model_from_json
from onset_detection import precision_times ,get_onsets, onsets_to_notes
from pitch_change import pitch_change
from find_bpm import get_bpm
import random

def generate(path, n_vocab, rating):

    #TODO: make this a funcion in onset_detection that returns the binary_onset_array
    onsets = get_onsets(path)
    bpm = get_bpm(path)
    notes, onsets  = onsets_to_notes(onsets, bpm, path,rating)
    num_32nd = len(precision_times(path, bpm))

    binary_onset_array = numpy.zeros(num_32nd + 1)
    for i in notes:
        binary_onset_array[i] = 1

    model = load_model()
    steps = generate_steps(model=model, n_vocab=n_vocab, onsets=onsets, path=path, bpm=bpm, rating=rating)
    stepchart = []
    j = 0
    for i in range(len(binary_onset_array)):
        b = binary_onset_array[i]
        if b == 0:
            line = [0,0,0,0]
            stepchart.append(line)
        else:
            if j >= len(steps):
                break
            line = decode_step(steps[j])
            stepchart.append(line)
            j+=1

    return stepchart


def load_model():
        # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    return loaded_model


def decode_step(num):
    """
    Convert int feature encoding to step line
    :param num: Int representing feature encoding
    :return: List of ints in [0, 2]
    """
    if num == 0:
        return [0, 0, 0, 0]
    step_line = []
    while num:
        num, r = divmod(num, 3)
        step_line.append(r)

    step_line += [0 for _ in range(4 - len(step_line))]
    return list(reversed(step_line))


def generate_steps(model, n_vocab, onsets, path, bpm, rating):
    """ Generate notes from the neural network based on a sequence of notes """

    prevNote = random.choice([1,3,9,27,27,27,27])
    steps = [prevNote]
    pitches = pitch_change(path,onsets,rating)
    print("Collected relative pitches")
    for i in range(1,len(pitches)):
        seed = random.randint(1, 20) #some randomness okay to prevent cycles/too much repetition in arrow types
        if seed < 7:
            prevNote = random.choice([1,3,9,27,27,27,27])
        fv = [prevNote, onsets[i] - onsets[i-1], onsets[i+1]-onsets[i] ,bpm, pitches[i-1]]

        prediction_input = numpy.reshape(fv, (1, 1, len(fv)))
        #print("prediction_input: ",prediction_input)
        prediction = model.predict(prediction_input, verbose=0)
        best_prediction = numpy.argmax(prediction)
        print("best_prediction", best_prediction)

        steps.append(best_prediction)
    print("Predicted steps")
    return steps


    # # generate steps for rest of song
    # for step_index, ele in enumerate(onsets):
    #     #print("element", step_index, ele, type(ele))
    #     #print("pattern Length:", len(pattern))
    #
    #     if int(ele) == 0:
    #         prediction_output.append([0,0,0,0])
    #         pattern.append(0)
    #         pattern = pattern[1:len(pattern)]
    #
    #     elif int(ele) == 1:
    #         prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
    #         prediction_input = prediction_input / float(n_vocab)
    #
    #         prediction = model.predict(prediction_input, verbose=0)
    #
    #         best_prediction = numpy.argmax(prediction)
    #         result = decode_step(best_prediction)
    #
    #         #print("result", step_index, result)
    #
    #         prediction_output.append(result)
    #
    #         pattern.append(best_prediction)
    #         pattern = pattern[1:len(pattern)]
    #
    # return prediction_output


if __name__ == '__main__':
    path = "../songs/sucker.wav"
    rating = "medium"
    n_vocab = 81
    stepchart = generate(path, n_vocab, rating)
    for step in stepchart:
        print(step)
