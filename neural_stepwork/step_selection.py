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
    difficulties = ["Hard","Medium","Challenge"]
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
            track = track['notes']
            if len(track) == 0:
                continue
            for line in track:
                line = ''.join(str(s) for s in line)
                y.append(step_to_int(line))
            y_train.append(y)
    print("finished loading training data\nnumber of charts = ",len(y_train))
    return y_train

# def construct_sequences_32nd_onsets(x_train, y_train, sequence_length =96, step=1):
#         # cut the text in semi-redundant sequences of sequence_length  words
#     sequences = []
#     next_step = []
#     ignored = 0
#
#     for track_idx in range(0, len(x_train)):
#         x_train_track = x_train[track_idx]
#         y_train_track = y_train[track_idx]
#
#         for i in range(0, len(x_train_track) - sequence_length , step):
#             sequences.append(x_train_track[i: i + sequence_length ])
#             next_step.append(y_train_track[i + sequence_length ])
#
#     return sequences, next_step



def step_to_int(step_line):
    line_to_int_dict = {'0000': 0, '0001': 1, '0002': 2, '0010': 3, '0011': 4, '0012': 5, '0020': 6, '0021': 7, '0022': 8, '0100': 9, '0101': 10, '0102': 11, '0110': 12, '0111': 13, '0112': 14, '0120': 15, '0121': 16, '0122': 17, '0200': 18, '0201': 19, '0202': 20, '0210': 21, '0211': 22, '0212': 23, '0220': 24, '0221': 25, '0222': 26, '1000': 27, '1001': 28, '1002': 29, '1010': 30, '1011': 31, '1012': 32, '1020': 33, '1021': 34, '1022': 35, '1100': 36, '1101': 37, '1102': 38, '1110': 39, '1111': 40, '1112': 41, '1120': 42, '1121': 43, '1122': 44, '1200': 45, '1201': 46, '1202': 47, '1210': 48, '1211': 49, '1212': 50, '1220': 51, '1221': 52, '1222': 53, '2000': 54, '2001': 55, '2002': 56, '2010': 57, '2011': 58, '2012': 59, '2020': 60, '2021': 61, '2022': 62, '2100': 63, '2101': 64, '2102': 65, '2110': 66, '2111': 67, '2112': 68, '2120': 69, '2121': 70, '2122': 71, '2200': 72, '2201': 73, '2202': 74, '2210': 75, '2211': 76, '2212': 77, '2220': 78, '2221': 79, '2222': 80}
    # line_as_string = ""
    # for digit in step_line:
    #     line_as_string = line_as_string + str(digit)
    #line_as_string = ''.join(str(s) for s in step_line)

    #TODO: This is temporarily here until the parser is fixed.
    # This is forcing all invalid step_lines to map to 1.
    if step_line in line_to_int_dict:
        line_as_int = line_to_int_dict[step_line]
    else:
        line_as_int = 1

    return line_as_int


def train_network():
    """ Train a Neural Network to generate music """
    n_vocab = vocab_size()

    network_input, network_output = prepare_sequences(n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def vocab_size():
    y_train = load_training_data()
    s = set()
    for y in y_train:
        for note in y:
            s.add(note)
    return len(list(s))

def prepare_sequences(n_vocab):
    sequence_length = 100
    step = 1
    y_train = load_training_data()[:5]

    network_input  = []
    network_output  = []

    #TODO: temporarily restricting tracks that are sequenced (this should iterate over all tracks that we have). It is currently just taking the first one
    for track_idx in range(len(y_train)):
        y_train_track = y_train[track_idx]

        for i in range(0, len(y_train_track) - sequence_length , step):
            sequence_in = y_train_track[i:i + sequence_length]
            sequence_out = y_train_track[i + sequence_length]
            network_input.append([step_to_int(step_line) for step_line in sequence_in])
            network_output.append(step_to_int(sequence_out))

    n_patterns = len(network_input)
    print("num patterns = ",n_patterns)
    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    #TODO: This is hardcoded in to force the one-hot array to create correct dimensions
    # What really needs to happen is we need to fix the parser and pass the right value for n_vocab around.
    network_output[0] = n_vocab - 1

    network_output = np_utils.to_categorical(network_output)
    print("finished preparing sequences")
    return (network_input, network_output)


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True))
    #model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    #model.add(LSTM(512))
    model.add(Dense(64))
    #model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print("finished making model\n")
    return model

def generate_random_sequence():
    mapping = {0:1,1:3,2:9,3:27}
    seq = np.zeros(100)
    for i in range(len(seq)):
        seq[i] = mapping[random.randint(0,3)]

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    print("starting to fit model")
    model.fit(network_input, network_output, epochs=1, batch_size=1600, callbacks=callbacks_list)
    print("finished fitting model")
    model.save('my_model.h5')
    print("starting to predict")
    prediction = model.predict(np.array(generate_random_sequence()))
    print("prediction: ",prediction)




if __name__ == '__main__':

    train_network()
