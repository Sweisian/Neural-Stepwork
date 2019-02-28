import os
from find_bpm import get_bpm
from onset_detection import get_onsets, notes_to_measures, onsets_to_notes


def write_simfile(title="", artist="", music="", offset=-0.000000, bpms=0):
    song_metadata = "#TITLE:{0};\n#ARTIST:{1};\n#MUSIC:{2};\n#OFFSET:{3};\n#BPMS:0.0={4};\n#STOPS:;\n".format(
        title, artist, music, offset, bpms
    )
    # setting difficulty level to 1
    # setting difficulty mode to edit (instead of hard, expert, etc.)
    # setting play mode to dance-singles
    # the series of zeros represents groove radar values; these values aren't required for now
    chart_metadata = (
        "#NOTES:\n\tdance-single:\n\t:\n\tEdit:\n\t1:\n\n0.0,0.0,0.0,0.0,0.0:\n"
    )
    # measures = ""
    # for i in range(30): #for now im saying there are 30 measures in the song
    #    measures += (generateRandomMeasure() + ",\n") #measures separated by commas
    # measures=measures[:-2] #remove last comma and new line character
    notes = onsets_to_notes(get_onsets(music), bpms)
    measures = notes_to_measures(notes, bpms, music)
    measures = measures.replace(".", "")
    measures = measures.replace(" ", "")
    simfile_content = song_metadata + chart_metadata + measures

    with open(title + ".sm", "w") as f:
        f.write(simfile_content)


def generate_random_measure():
    return "1000\n0100\n1000\n0010\n0100\n0001\n0010\n0100\n"


music = "songs/Nekozilla.wav"
write_simfile(title="Nekozilla", music=music, bpms=get_bpm(music))
