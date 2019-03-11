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
    DATA_DIR = "../training/json"
    y_train = list()
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(DATA_DIR, file)) as f:
            step_file = json.load(f)
        for track in step_file["notes"]:
            y = []
            if track["difficulty_coarse"] not in difficulties:
                continue
            track = track["notes"]
            if len(track) == 0:
                continue
            for line in track:
                y.append(encode_step(line))
            y_train.append(y)
    print("finished loading training data\nnumber of charts = ", len(y_train))

    return y_train, encode_step([2, 2, 2, 2]) + 1


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
    y_train = [y_train[0]]

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

    for track in y_train:
        for window_start in range(0, len(track) - sequence_length):
            if track[window_start + sequence_length] == 0:
                continue
            sequence_in = track[window_start : window_start + sequence_length]
            sequence_out = track[window_start + sequence_length]
            network_input.append(sequence_in)
            network_output.append(sequence_out)

    n_patterns = len(network_input)
    print("num patterns = ", n_patterns)
    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output, num_classes=n_vocab)
    print("finished preparing sequences")
    return network_input, network_output


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
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
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
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
        epochs=1,
        batch_size=1600,
        callbacks=callbacks_list,
    )
    print("finished fitting model")
    # model.save("my_model.h5")
    print("starting to predict")
    prediction = model.predict(generate_random_sequence())
    print("prediction: ", decode_step(np.argmax(prediction)))


if __name__ == "__main__":

    train_network()
