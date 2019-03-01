from .onset_detection import get_onsets, notes_to_measures, onsets_to_notes


def write_simfile(dest, title="", artist="", music="", offset=-0.000000, bpms=0):
    #TODO: update offset to reflect silence in beginning of song

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

    #convert onsets (in seconds) to notes (the nth 16th note in the song)
    notes = onsets_to_notes(get_onsets(music), bpms, music)
    #given a series of notes, create a series of measures (1 where there is a note, 0 where there isnt)
    #for now, we randomly decide which arrow type (up, down, etc.) to use
    measures = notes_to_measures(notes, bpms, music)
    measures = measures.replace(".", "")
    measures = measures.replace(" ", "")
    simfile_content = song_metadata + chart_metadata + measures

    #write notes and metadata to a simfile
    with open(dest + title + ".sm", "w") as f:
        f.write(simfile_content)
