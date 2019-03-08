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

def construct_sentences(x_train, y_train, SEQUENCE_LEN=96, STEP=1):
        # cut the text in semi-redundant sequences of SEQUENCE_LEN words
    sentences = []
    next_step = []
    ignored = 0

    for track_idx in range(0, len(x_train)):
        x_train_track = x_train[track_idx]
        y_train_track = y_train[track_idx]

        for i in range(0, len(x_train_track) - SEQUENCE_LEN, STEP):
            sentences.append(x_train_track[i: i + SEQUENCE_LEN])
            next_step.append(y_train_track[i + SEQUENCE_LEN])

    return sentences, next_step

def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=2):
    # shuffle at unison
    print('Shuffling sentences')

    tmp_sentences = []
    tmp_next_word = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    print("Size of training set = %d" % len(x_train))
    print("Size of test set = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)


def step_to_int(y_train):

    y_train_flat = np.zeros((0))

    for track in y_train:
        for step in track:
            np.append(y_train_flat, step)

    print(len(y_train_flat))
    print(type(y_train_flat))

    # test_list = [1,2,3,4]
    # print(type(test_list))
    #
    # step_set = set(test_list)
    #
    # for i in range(0,100):
    #     print(y_train_flat[i])
    #     print(type(y_train_flat[i]))


    step_types = sorted(set(y_train_flat))

    print("step Types",step_types)

    step_to_int = dict((c, i) for i, c in enumerate(step_types))
    int_to_step = dict((i, c) for i, c in enumerate(step_types))
    return  step_to_int, int_to_step


x_train, y_train = load_training_data()

step_to_int, int_to_step = step_to_int(y_train)


sentences, next_words, sentences_test, next_words_test = shuffle_and_split_training_set(sentences, next_words)


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
