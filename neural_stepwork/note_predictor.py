import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation

def generate(onset_array, n_vocab):
    #TODO: handle the step-to-int mapping in some way.
    model = load_model()
    prediction_output = generate_steps(model, n_vocab, onset_array)
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
    pattern = np.zeros(100)
    prediction_output = []

    # generate steps for rest of song
    for step_index, ele in enumerate(onsets):
        if ele == 0:
            prediction_output.append([0,0,0,0])
            pattern.append(0)
            pattern = pattern[1:len(pattern)]

        elif ele == 1:
            #TODO: need to reshape before predicting
            prediction_input = pattern

            prediction = model.predict(prediction_input, verbose=0)

            best_prediction = numpy.argmax(prediction)
            result = decode_step(best_prediction)
            prediction_output.append(result)

            norm_best_pred = best_prediction / float(n_vocab)
            pattern.append(norm_best_pred)
            pattern = pattern[1:len(pattern)]

    return prediction_output

#if __name__ == '__main__':
    #generate()
