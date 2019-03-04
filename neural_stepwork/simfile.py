from .onset_detection import get_onsets, notes_to_measures, onsets_to_notes

def write_simfile(source, dest, name, bpm, artist="Unknown Artist", offset=-0.000000):
    song_metadata = "#TITLE:{0};\n#ARTIST:{1};\n#MUSIC:{2};\n#OFFSET:{3};\n#BPMS:0.0={4};\n#STOPS:;\n".format(
        name, artist, source, offset, bpm
    )
    # setting difficulty level to 1
    # setting difficulty mode to edit (instead of hard, expert, etc.)
    # setting play mode to dance-singles
    # the series of zeros represents groove radar values; these values aren't required for now
    chart_metadata = "#NOTES:\n\tdance-single:\n\t:\n\tEdit:\n\t1:\n\n0.0,0.0,0.0,0.0,0.0:\n"
    notes = onsets_to_notes(get_onsets(source), bpm, source)
    measures = notes_to_measures(notes, bpm, source)
    measures = measures.replace(".", "")
    measures = measures.replace(" ", "")
    simfile_content = song_metadata + chart_metadata + measures

    with open(dest + name + ".sm", "w") as f:
        f.write(simfile_content)


