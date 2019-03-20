import numpy
from .onset_detection import precision_times ,get_onsets, onsets_to_notes
from .step_selection import predict_decision_tree, train_decision_tree, load_decision_tree
import random
from .pitch_change import pitch_change


def generate(path,bpm,rating):
    """
    :param path: path to audio file
    :param bpm: tempo of song
    :param rating: difficulty rating of chart (eays, medium, hard)
    :return: list of lists representing the lines in a stepchart
    """
    onsets = get_onsets(path)
    notes, onsets  = onsets_to_notes(onsets, bpm, path,rating)
    print("Detected onsets for given difficulty rating")
    num_32nd = len(precision_times(path, bpm))
    binary_onset_array = numpy.zeros(num_32nd + 1)
    for i in notes:
        binary_onset_array[i] = 1
    steps = generate_dt(path,onsets,bpm,rating)
    stepchart = []
    j = 0
    for i in range(len(binary_onset_array)):
        b = binary_onset_array[i]
        if b == 0:
            line = [0,0,0,0]
            stepchart.append(line)
        else:
            if j >= len(steps):
                break
            line = decode_step(steps[j])
            stepchart.append(line)
            j+=1
    return stepchart

def generate_dt(path,onsets,bpm,rating):
    """
    :param path: file path to audio file
    :param onsets: list of onsets in seconds
    :param bpm: tempo of audio
    :param rating: difficulty rating of chart (easy, medium, hard)
    :return: list of integers representing step selection for a chart
    """
    dt = load_decision_tree()
    print("Loaded decision tree classifier")
    prevNote = random.randint(1,80)
    steps = [prevNote]
    pitches = pitch_change(path,onsets,rating)
    print("Collected relative pitches")
    for i in range(1,len(pitches)):
        seed = random.randint(1, 20) #some randomness okay to prevent cycles/too much repetition in arrow types
        if seed < 7:
            prevNote = random.choice([1,3,9,27,27,27,27])
        fv = [prevNote,onsets[i] - onsets[i-1],onsets[i+1]-onsets[i],bpm,pitches[i-1]]
        prevNote = predict_decision_tree(dt,fv)[0]
        steps.append(prevNote)
    print("Predicted steps")
    return steps


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

<<<<<<< HEAD

def generate_steps(model, n_vocab, onsets):
    """ Generate notes from the neural network based on a sequence of notes """

    #Start with 100 zeros
    pattern = [0]
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


=======
>>>>>>> 50031d223dba5ccdc7957596badda8c931b6e69c
if __name__ == '__main__':
    pass
    #path = "../songs/sucker.wav"
    #print(generate(path))
