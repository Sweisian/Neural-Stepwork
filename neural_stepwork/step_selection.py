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

    return y_train, encode_step([1, 1, 1, 1]) + 1


def encode_step(step_line):
    """
    Convert step line to int
    :param step_line: List of ints in [0, 2]
    :return: Int representing feature encoding
    """
    step_line = [1 if step == 2 else step for step in step_line]
    return int("".join(str(x) for x in step_line), base=2)


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
        num, r = divmod(num, 2)
        step_line.append(r)

    step_line += [0 for _ in range(4 - len(step_line))]
    return list(reversed(step_line))


def train_network():
    """ Train a Neural Network to generate music """
    y_train, n_vocab = load_training_data()

    y_train = y_train[:1]

    network_input, network_output = prepare_sequences(y_train, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def prepare_sequences(data, n_vocab, sequence_length=10):
    """
    Create input sequences and their outputs for the model, making sure that each sequence
    ends with a note that has at least one arrow
    :param data: List of list of its representing notes
    :param n_vocab: Number of different possible notes
    :param sequence_length: Number of notes to include in each sequence
    :return: List of lists of notes (as ints), list of corresponding following notes
    """
    network_input = []
    network_output = []

    cleaned_data = []
    for track in data:
        cleaned_track = []
        for note in track:
            if note == 0:
                continue
            categorical_note = np_utils.to_categorical(note, num_classes=n_vocab)
            cleaned_track.append(categorical_note)
        cleaned_data.append(cleaned_track)

    for track in cleaned_data:
        for window_start in range(0, len(track) - sequence_length):
            sequence_in = track[window_start : window_start + sequence_length]
            sequence_out = track[window_start + sequence_length]
            network_input.append(sequence_in)
            network_output.append(sequence_out)

    n_patterns = len(network_input)
    print("num patterns = ", n_patterns)
    # reshape the input into a format compatible with LSTM layers
    # network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    return np.array(network_input), np.array(network_output)


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(
        LSTM(
            128,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True,
        )
    )
    model.add(LSTM(128))
    #model.add(Dense(64))
    model.add(Dense(n_vocab))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    print("finished making model")
    return model


def generate_random_sequence(n_vocab, seq_len):

    seq = []
    for i in range(seq_len):
        seq.append(np_utils.to_categorical(random.randint(0, 4), num_classes=n_vocab))
    return seq


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
        epochs=100,
        batch_size=16,
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
    n_vocab = 16
    seq_len = 10
    y_train, n_vocab = load_training_data()
    network_input, network_output = prepare_sequences(y_train, n_vocab)
    model = create_network(network_input, n_vocab)
    model.load_weights("model.h5")
    print("starting to predict")
    seq = generate_random_sequence(n_vocab, seq_len)
    print(seq)
    for i in range(100):
        reshaped = np.reshape(seq, (1, seq_len, n_vocab))
        prediction = model.predict(reshaped)
        #print(prediction)
        encoded_step = np.argmax(prediction)
        print(decode_step(encoded_step))
        next_step = np_utils.to_categorical(encoded_step, num_classes=n_vocab)
        # print(next_step)
        seq = seq[1:] + [next_step]

if __name__ == "__main__":
    train_network()
    load_trained_model()

