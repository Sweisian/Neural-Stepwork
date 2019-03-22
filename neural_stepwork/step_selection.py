import os
import json
<<<<<<< HEAD
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from pitch_change import pitch_change

import random
=======
from .pitch_change import pitch_change
import pickle
from sklearn.tree import DecisionTreeClassifier
>>>>>>> fcdd3ad8f08121d8b6d41fe47abc040253377d66

def load_training_data():
    """
    :return: y_train, which is a list (each song) of lists (every onset in song) of lists (features at each onset)
             **first element in each song's list is song bpm
    """
<<<<<<< HEAD
    difficulties = ["hard", "medium", "challenge"]
    #cwd = os.getcwd()
    #DATA_DIR = cwd + "/training/json"
=======
<<<<<<< HEAD
    difficulties = ["Hard", "Medium", "Challenge"]
#    cwd = os.getcwd()
#    DATA_DIR = cwd + "/training/json"
=======
    difficulties = ["hard", "medium", "challenge"]
    #cwd = os.getcwd()
    #DATA_DIR = cwd + "/training/json"
>>>>>>> 50031d223dba5ccdc7957596badda8c931b6e69c
>>>>>>> fcdd3ad8f08121d8b6d41fe47abc040253377d66
    DATA_DIR =  "../training/json"
    y_train = list()
    #print("i got here too")
    for file in os.listdir(DATA_DIR)[:20]:
        #print("get here")
        if not file.endswith(".json"):
            continue
        with open(os.path.join(DATA_DIR, file)) as f:
            step_file = json.load(f)
        for track in step_file["notes"]:
<<<<<<< HEAD
            rating = track["difficulty_coarse"].lower()
            if rating not in difficulties:
=======
<<<<<<< HEAD
            if track["difficulty_coarse"] not in difficulties:
=======
            rating = track["difficulty_coarse"].lower()
            if rating not in difficulties:
>>>>>>> 50031d223dba5ccdc7957596badda8c931b6e69c
>>>>>>> fcdd3ad8f08121d8b6d41fe47abc040253377d66
                continue
            track = track["notes"]
            if len(track) == 0:
                continue
            bpms = step_file['bpms']
            y = [maxBpm(bpms)]
            lineNum = 0
            priorTime = 0
<<<<<<< HEAD
            times = []
=======
<<<<<<< HEAD
=======
            times = []
>>>>>>> 50031d223dba5ccdc7957596badda8c931b6e69c
>>>>>>> fcdd3ad8f08121d8b6d41fe47abc040253377d66
            for line in track:
                step = encode_step(line)
                if step != 0:
                    time = lineNumToTime(bpms,priorTime)
<<<<<<< HEAD
                    times.append(time)
=======
<<<<<<< HEAD
>>>>>>> fcdd3ad8f08121d8b6d41fe47abc040253377d66
                    priorTime = time
                    y+=[[time,step,0]]
                lineNum += 1
            name = os.path.basename(file).split(".json")[0]
            path = "../training/raw" + "/" + name + "/" + name + ".wav"
            print(52,path)
            try:
                pitches = pitch_change(path, times,rating)
                for i in range(1,len(pitches)):
                    y[i][2]=pitches[i]
                y_train.append(y)
            except:
                print("I excepted out")
                continue
    print("finished loading training data\nnumber of charts = ", len(y_train))
    return y_train, encode_step([2, 2, 2, 2]) + 1
=======
                    times.append(time)
                    priorTime = time
                    y+=[[time,step,0]]
                lineNum += 1
            name = os.path.basename(file).split(".json")[0]
            path = "../training/raw" + "/" + name + "/" + name + ".wav"
            print(52,path)
            try:
                pitches = pitch_change(path, times,rating)
                for i in range(1,len(pitches)):
                    y[i][2]=pitches[i]
                y_train.append(y)
            except:
                continue
    print("finished loading training data\nnumber of charts = ", len(y_train))
    return y_train


def lineNumToTime(bpms,priorTime):
    """
    :param bpms: list of lists (length 2) indicating at what times different bpms in the chart begin
    :param priorTime: time of last line in stepchart
    :return: time of next line in stepchart
    """
    precision = 32
    i = 0
    while priorTime < bpms[i][0]:
        i+=1
    bpm = bpms[i-1][1]
    per_second = bpm * (1 / 60.0) * (precision/4)
    increment = 1 / per_second
    return (priorTime + increment)


def maxBpm(bpms):
    """
    :param bpms: list of lists (length 2) indicating at what times different bpms in the chart begin
    :return: max bpm that occurs in a song
    """
    mx = 0.0
    for b in bpms:
        mx = max(mx,max(b))
    return mx
>>>>>>> 50031d223dba5ccdc7957596badda8c931b6e69c

# def load_training_data():
#     """
#     y_train is a list of lists, and each list is a simfile chart as a series of ints (each int maps to a possible step line)
#     :return: y_train
#     """
#     difficulties = ["Hard", "Medium", "Challenge"]
# #    cwd = os.getcwd()
# #    DATA_DIR = cwd + "/training/json"
#     DATA_DIR =  "../training/json"
#     y_train = list()
#     for file in os.listdir(DATA_DIR):
#         if not file.endswith(".json"):
#             continue
#         with open(os.path.join(DATA_DIR, file)) as f:
#             step_file = json.load(f)
#         for track in step_file["notes"]:
#             if track["difficulty_coarse"] not in difficulties:
#                 continue
#             track = track["notes"]
#             if len(track) == 0:
#                 continue
#             bpms = step_file['bpms']
#             y = [maxBpm(bpms)]
#             lineNum = 0
#             priorTime = 0
#             for line in track:
#                 step = encode_step(line)
#                 if step != 0:
#                     time = lineNumToTime(bpms,priorTime)
#                     priorTime = time
#                     y.append((time,step))
#                 lineNum += 1
#             y_train.append(y)
#     print("finished loading training data\nnumber of charts = ", len(y_train))
#     return y_train, encode_step([2, 2, 2, 2]) + 1


def lineNumToTime(bpms,priorTime):
    """
    :param bpms: list of lists (length 2) indicating at what times different bpms in the chart begin
    :param priorTime: time of last line in stepchart
    :return: time of next line in stepchart
    """
    precision = 32
    i = 0
    while priorTime < bpms[i][0]:
        i+=1
    bpm = bpms[i-1][1]
    per_second = bpm * (1 / 60.0) * (precision/4)
    increment = 1 / per_second
    return (priorTime + increment)


def maxBpm(bpms):
    """
    :param bpms: list of lists (length 2) indicating at what times different bpms in the chart begin
    :return: max bpm that occurs in a song
    """
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


def prepare_sequences(y_train):
    """
    Create input sequences and their outputs for the model, making sure that each sequence
    ends with a note that has at least one arrow
    :param y_train: List of list of its representing notes
    :return: X: List of lists (Feature vectors) for onsets in songs/charts
             y: integer representation of next step in the chart
    """
    X = []
    y = []

<<<<<<< HEAD
    #TODO: Uncomment this after testing
=======
>>>>>>> 50031d223dba5ccdc7957596badda8c931b6e69c
    #for track in y_train:
    for track in y_train:
        #Only doing these amounts to not get bpm (first element) and make math work for first and last elements (cant get diff for those)
        for idx, time_step_pair in enumerate(track[:-1]):
            #skip bpm (first element) and first tuple so math works
            if idx == 0 or idx == 1:
                continue
<<<<<<< HEAD

<<<<<<< HEAD
=======

=======
>>>>>>> 50031d223dba5ccdc7957596badda8c931b6e69c
>>>>>>> fcdd3ad8f08121d8b6d41fe47abc040253377d66
            bpm = track[0]
            time_to_prev_step = track[idx][0] - track[idx - 1][0]
            time_to_next_step = track[idx + 1][0] - track[idx][0]
            prev_note = track[idx - 1][1]
            curr_note = track[idx][1]
<<<<<<< HEAD
            relativePitch = track[idx][2]
            feature_vect = [prev_note, time_to_prev_step, time_to_next_step, bpm, relativePitch]
            #feature_vect = [prev_note, time_to_prev_step, time_to_next_step, bpm]
=======
<<<<<<< HEAD
            feature_vect = [prev_note, time_to_prev_step, time_to_next_step, bpm]
>>>>>>> fcdd3ad8f08121d8b6d41fe47abc040253377d66
            network_input.append(feature_vect)
            network_output.append(curr_note)

    # reshape input array
    network_input = np.asarray(network_input)
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
        epochs=25,
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
=======
            relativePitch = track[idx][2]
            feature_vect = [prev_note, time_to_prev_step, time_to_next_step, bpm,relativePitch]
            X.append(feature_vect)
            y.append(curr_note)
    return X, y


def load_decision_tree():
    """
    :return: decision tree model loaded from file to avoid re-training every time
    """
    decision_tree_pkl_filename = 'decision_tree_classifier.pkl'
    decision_tree_model_pkl = open(decision_tree_pkl_filename, 'rb')
    decision_tree_model = pickle.load(decision_tree_model_pkl)
    return decision_tree_model
>>>>>>> 50031d223dba5ccdc7957596badda8c931b6e69c

def train_decision_tree():
    """
    train decision free on onset feature vectors and save to pickle file
    """

    y_train= load_training_data()
    X, y = prepare_sequences(y_train)
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=200, min_samples_leaf=5)
    clf_entropy.fit(X, y)

<<<<<<< HEAD
    train_network()
    #load_trained_model()
=======
    # Dump the trained decision tree classifier with Pickle
    decision_tree_pkl_filename = 'decision_tree_classifier.pkl'
    # Open the file to save as pkl file
    decision_tree_model_pkl = open(decision_tree_pkl_filename, 'wb')
    pickle.dump(clf_entropy, decision_tree_model_pkl)
    # Close the pickle instances
    decision_tree_model_pkl.close()

def predict_decision_tree(dt,fv):
    """
    :param dt: decision tree model
    :param fv: feature vector
    :return: integer representation of predicted next step in chart
    """
    return dt.predict([fv])



if __name__ == "__main__":
    train_decision_tree()
>>>>>>> 50031d223dba5ccdc7957596badda8c931b6e69c
