import numpy as np
import os
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import random


def load_training_data():
    """
    y_train is a list of lists, and each list is a simfile chart as a series of ints (each int maps to a possible step line)
    :return: y_train
    """
    difficulties = ["Hard", "Medium", "Challenge"]
#    cwd = os.getcwd()
#    DATA_DIR = cwd + "/training/json"
    DATA_DIR =  "../training/json"
    y_train = list()
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(DATA_DIR, file)) as f:
            step_file = json.load(f)
        for track in step_file["notes"]:
            if track["difficulty_coarse"] not in difficulties:
                continue
            track = track["notes"]
            if len(track) == 0:
                continue
            bpms = step_file['bpms']
            y = [maxBpm(bpms)]
            lineNum = 0
            priorTime = 0
            for line in track:
                step = encode_step(line)
                if step != 0:
                    time = lineNumToTime(bpms,priorTime)
                    priorTime = time
                    y.append((time,step))
                lineNum += 1
            y_train.append(y)
    print("finished loading training data\nnumber of charts = ", len(y_train))
    return y_train, encode_step([2, 2, 2, 2]) + 1


def lineNumToTime(bpms,priorTime):
    precision = 32
    i = 0
    while priorTime < bpms[i][0]:
        i+=1
    bpm = bpms[i-1][1]
    per_second = bpm * (1 / 60.0) * (precision/4)
    increment = 1 / per_second
    return (priorTime + increment)


def maxBpm(bpms):
    mx = 0.0
    for b in bpms:
        mx = max(mx,max(b))
    return mx


def encode_step(step_line):
    """
    Convert step line to int
    :param step_line: List of ints in [0, 2]
    :return: Int representing feature encoding
    """
    return int("".join(str(x) for x in step_line), base=3)


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


def train_network():
    """ Train a Neural Network to generate music """
    y_train, n_vocab = load_training_data()

    network_input, network_output = prepare_sequences(y_train, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def prepare_sequences(y_train, n_vocab, sequence_length=100):
    """
    Create input sequences and their outputs for the model, making sure that each sequence
    ends with a note that has at least one arrow
    :param y_train: List of list of its representing notes
    :param n_vocab: Number of different possible notes
    :param sequence_length: Number of notes to include in each sequence
    :return: List of lists of notes (as ints), list of corresponding following notes
    """
    network_input = []
    network_output = []

    #TODO: Uncomment this after testing
    #for track in y_train:
    for track in y_train:
        #Only doing these amounts to not get bpm (first element) and make math work for first and last elements (cant get diff for those)
        for idx, time_step_pair in enumerate(track[:-1]):
            #skip bpm (first element) and first tuple so math works
            if idx == 0 or idx == 1:
                continue


            bpm = track[0]
            time_to_prev_step = track[idx][0] - track[idx - 1][0]
            time_to_next_step = track[idx + 1][0] - track[idx][0]
            prev_note = track[idx - 1][1]
            curr_note = track[idx][1]
            feature_vect = [prev_note, time_to_prev_step, time_to_next_step, bpm]
            network_input.append(feature_vect)
            network_output.append(curr_note)

    # normalize input
    network_input = np.asarray(network_input)
    network_input = network_input / network_input.max(axis=0)
    network_input = network_input.reshape(len(network_input), 1, len(network_input[0]))

    # network_input_normed = np.tolist(network_input_normed)

    network_output = np_utils.to_categorical(network_output, num_classes=n_vocab)
    print("finished preparing sequences")
    # print(network_output[0])
    # print(network_output[1])
    # print(network_output[2])
    print(network_input)
    print(network_input[0])

    return network_input, network_output


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """

    #print("Network input shape",network_input.shape[0],network_input.shape[1], network_input.shape[2])

    model = Sequential()
    model.add(
        LSTM(
            512,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True,
        )
    )
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True))
    # model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    # model.add(LSTM(512))
    model.add(Dense(64))
    # model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['acc'])
    print("finished making model\n")
    return model


def generate_random_sequence():
    mapping = {0: 1, 1: 3, 2: 9, 3: 27}
    seq = np.zeros(100)
    for i in range(len(seq)):
        seq[i] = mapping[random.randint(0, 3)]
    return np.reshape(seq, (1, 100, 1))


def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor="loss", verbose=0, save_best_only=True, mode="min"
    )
    callbacks_list = [checkpoint]
    print("starting to fit model")
    model.fit(
        network_input,
        network_output,
        epochs=30,
        batch_size=64,
        callbacks=callbacks_list,
    )
    print("finished fitting model")

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")



def load_trained_model():
    y_train, n_vocab = load_training_data()
    network_input, network_output = prepare_sequences(y_train, n_vocab)
    model = create_network(network_input, n_vocab)
    model.load_weights("../neural_stepwork/my_model.h5")
    print("starting to predict")
    for i in range(50):
        seq = generate_random_sequence()
        seq = seq/(n_vocab-1)
        prediction = model.predict(seq)
        print(decode_step(np.argmax(prediction)))

if __name__ == "__main__":

    train_network()
    #load_trained_model()
