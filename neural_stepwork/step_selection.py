import os
import json
from pitch_change import pitch_change
import pickle
from sklearn.tree import DecisionTreeClassifier


def load_training_data():
    """
    :return: y_train, which is a list (each song) of lists (every onset in song) of lists (features at each onset)
             **first element in each song's list is song bpm
    """
    difficulties = ["hard", "medium", "challenge"]
    #cwd = os.getcwd()
    #DATA_DIR = cwd + "/training/json"
    DATA_DIR =  "../training/json"
    y_train = list()
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(DATA_DIR, file)) as f:
            step_file = json.load(f)
        for track in step_file["notes"]:
            rating = track["difficulty_coarse"].lower()
            if rating not in difficulties:
                continue
            track = track["notes"]
            if len(track) == 0:
                continue
            bpms = step_file['bpms']
            y = [maxBpm(bpms)]
            lineNum = 0
            priorTime = 0
            times = []
            for line in track:
                step = encode_step(line)
                if step != 0:
                    time = lineNumToTime(bpms,priorTime)
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

def train_decision_tree():
    """
    train decision free on onset feature vectors and save to pickle file
    """

    y_train= load_training_data()
    X, y = prepare_sequences(y_train)
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=200, min_samples_leaf=5)
    clf_entropy.fit(X, y)

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