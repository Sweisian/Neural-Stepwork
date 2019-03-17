import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.models import model_from_json
from onset_detection import times16ths ,get_onsets, notes_to_measures, onsets_to_notes
from find_bpm import get_bpm

def generate(binary_onset_array, n_vocab):
    #TODO: handle the step-to-int mapping in some way.
    model = load_model()
    prediction_output = generate_steps(model, n_vocab, binary_onset_array)

    return prediction_output


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


def generate_steps(model, n_vocab, onsets):
    """ Generate notes from the neural network based on a sequence of notes """

    #Start with 100 zeros
    pattern = [0] * 100
    prediction_output = []

    # generate steps for rest of song
    for step_index, ele in enumerate(onsets):
        #print("element", step_index, ele, type(ele))
        #print("pattern Length:", len(pattern))

        if int(ele) == 0:
            prediction_output.append([0,0,0,0])
            pattern.append(0)
            pattern = pattern[1:len(pattern)]

        elif int(ele) == 1:
            #TODO: need to reshape before predicting
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)

            prediction = model.predict(prediction_input, verbose=0)

            best_prediction = numpy.argmax(prediction)
            result = decode_step(best_prediction)

            #print("result", step_index, result)

            prediction_output.append(result)

            pattern.append(best_prediction)
            pattern = pattern[1:len(pattern)]

    return prediction_output




if __name__ == '__main__':
    source = "../songs/sucker.wav"
    n_vocab = 81
    onsets = get_onsets(source)
    bpm = get_bpm(source)
    notes = onsets_to_notes(onsets, bpm, source)
    num_16ths = len(times16ths(source, bpm))
    binary_onset_array = numpy.zeros(num_16ths + 1)
    for i in notes:
        binary_onset_array[i] = 1
    #print(notes)

    print(generate(binary_onset_array, n_vocab))
