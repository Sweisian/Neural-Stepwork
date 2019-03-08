import numpy as np
import os
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM


def load_training_data():
    """
    x_train should be a 2D 2000 by n array where n is number of training examples
    y_train should be a 3D 2000 by 4 by n array where n is number of training examples
    :return: x_train, y_train
    """
    DATA_DIR = "../training/json"
    y_train = list()
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(DATA_DIR, file)) as f:
            step_file = json.load(f)
        for track in step_file["notes"]:
            y_train.append(np.asarray(track["notes"]))
    x_train = [[1 if any(note) else 0 for note in example] for example in y_train]
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    return x_train, y_train

# for e in y_train[0]:
#     print(e)

# def construct_sequences_32nd_onsets(x_train, y_train, SEQUENCE_LEN=96, STEP=1):
#         # cut the text in semi-redundant sequences of SEQUENCE_LEN words
#     sequences = []
#     next_step = []
#     ignored = 0
#
#     for track_idx in range(0, len(x_train)):
#         x_train_track = x_train[track_idx]
#         y_train_track = y_train[track_idx]
#
#         for i in range(0, len(x_train_track) - SEQUENCE_LEN, STEP):
#             sequences.append(x_train_track[i: i + SEQUENCE_LEN])
#             next_step.append(y_train_track[i + SEQUENCE_LEN])
#
#     return sequences, next_step


def construct_sequences_just_step_lines(y_train, SEQUENCE_LEN=96, STEP=1):
    network_input  = []
    network_output  = []

    #TODO: temporarily restricting tracks that are sequenced
    for track_idx in range(0, 3):
        y_train_track = y_train[track_idx]

        for i in range(0, len(y_train_track) - SEQUENCE_LEN, STEP):
            sequence_in = y_train_track[i:i + SEQUENCE_LEN]
            sequence_out = y_train_track[i + SEQUENCE_LEN]
            network_input.append(steps_to_int(sequence_in))
            #print("Done with seq in", steps_to_int(sequence_in))
            network_output.append(steps_to_int([sequence_out]))
            #print("Done with seq out", steps_to_int([sequence_out]))
    return network_input, network_output


def steps_to_int(step_lines):
    line_to_int = {'0000': 0, '0001': 1, '0002': 2, '0010': 3, '0011': 4, '0012': 5, '0020': 6, '0021': 7, '0022': 8, '0100': 9, '0101': 10, '0102': 11, '0110': 12, '0111': 13, '0112': 14, '0120': 15, '0121': 16, '0122': 17, '0200': 18, '0201': 19, '0202': 20, '0210': 21, '0211': 22, '0212': 23, '0220': 24, '0221': 25, '0222': 26, '1000': 27, '1001': 28, '1002': 29, '1010': 30, '1011': 31, '1012': 32, '1020': 33, '1021': 34, '1022': 35, '1100': 36, '1101': 37, '1102': 38, '1110': 39, '1111': 40, '1112': 41, '1120': 42, '1121': 43, '1122': 44, '1200': 45, '1201': 46, '1202': 47, '1210': 48, '1211': 49, '1212': 50, '1220': 51, '1221': 52, '1222': 53, '2000': 54, '2001': 55, '2002': 56, '2010': 57, '2011': 58, '2012': 59, '2020': 60, '2021': 61, '2022': 62, '2100': 63, '2101': 64, '2102': 65, '2110': 66, '2111': 67, '2112': 68, '2120': 69, '2121': 70, '2122': 71, '2200': 72, '2201': 73, '2202': 74, '2210': 75, '2211': 76, '2212': 77, '2220': 78, '2221': 79, '2222': 80}
    track_in_ints = np.zeros((0))
    for step_line in step_lines:
        line_as_string = ''

        #print(step_line)

        for char in step_line:
            line_as_string = line_as_string + str(char)

        #TODO: This is temporarily here until the parser is fixed.
        if line_as_string in line_to_int:
            track_in_ints = np.append(track_in_ints, [line_to_int[line_as_string]])
        else:
            track_in_ints = np.append(track_in_ints, [1])

    return  track_in_ints


x_train, y_train = load_training_data()
network_input, network_output = construct_sequences_just_step_lines(y_train)

print("network_input[0]: ", network_input)
print("network_output[0]: ", network_output)


#sequences, next_words, sequences_test, next_words_test = shuffle_and_split_training_set(sequences, next_words)


def train_model():
    (x_train,y_train) = load_training_data()
    #notes in a stepchart have sequential dependence
    model = Sequential()
    model.add(LSTM(256,input_shape=(x_train.shape[0],1))) #2000 rows by 1 col (1 if note there, 0 if not)
    model.add(Dropout(0.2))
    model.add(Dense((y_train.shape[0],y_train.shape[1]), activation='softmax')) #2000 rows by 4 cols
    model.compile(loss='categorical_crossentropy', optimizer='adam') #TODO: not sure about this line ????
    model.fit(x_train, y_train, epochs=20, batch_size=1) # , callbacks=callbacks_list)
    #TODO: save model

def predict_with_model(model,notes,int_to_line):
    # pick a random seed
    start = np.random.randint(0, len(notes) - 1)
    pattern = notes[start]
    for i in range(2000):
        prediction = model.predict(notes, verbose=0)
        index = np.argmax(prediction)
        result = int_to_line[index]
        seq_in = [int_to_line[value] for value in pattern]
        pattern.append(index)
        pattern = pattern[1:len(pattern)]


def shuffle_and_split_training_set(sequences_original, next_original, percentage_test=2):
    # shuffle at unison
    print('Shuffling sequences')

    tmp_sequences = []
    tmp_next_word = []
    for i in np.random.permutation(len(sequences_original)):
        tmp_sequences.append(sequences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sequences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sequences[:cut_index], tmp_sequences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    print("Size of training set = %d" % len(x_train))
    print("Size of test set = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)
