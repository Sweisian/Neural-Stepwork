from numpy import array_split


def write_metadata(file, title, artist, source, offset, bpm):
    """
    Writes chart metadata to file
    :param file: File to write to
    :param title: Title of the track
    :param artist: Name of the track's artist
    :param source: Source audio file
    :param offset: Time of the track (in seconds) to start at
    :param bpm: BPM of the track
    """
    metadata = (
        f"#TITLE:{title};\n"
        f"#ARTIST:{artist};\n"
        f"#MUSIC:{source};\n"
        f"#OFFSET:{offset};\n"
        f"#BPMS:0.0={bpm};\n"
    )

    with open(file, "a") as sm:
        sm.write(metadata)


def write_notes_header(file, creator="Neural-Stepwork", rating="Edit", difficulty=1):
    """
    Writes note header to file
    :param file: File to write to
    :param creator: Source of the stepfile
    :param rating: Rough description of difficulty
    :param difficulty: Numeric description of difficulty
    """
    # Zeroes represent groove radar values, which are currently unused
    header = (
        "#NOTES:\n"
        "\tdance-single:\n"
        f"\t{creator}:\n"
        f"\t{rating}:\n"
        f"\t{difficulty}:\n"
        f"\t0.0,0.0,0.0,0.0,0.0:\n"
    )

    with open(file, "a") as sm:
        sm.write(header)


def write_notes(file, notes, measure_length=32):
    """
    Writes measures, separated by commas, to file
    :param file: File to write to
    :param notes: List-like with shape (num_notes, 4)
    :param measure_length: Number of notes to include in a measure
    """
    # Pad notes to be a multiple of measure_length
    if len(notes) % measure_length != 0:
        for _ in range(measure_length - (len(notes) % measure_length)):
            notes.append([0, 0, 0, 0])

    measures = array_split(notes, len(notes) / measure_length)
    note_format = "{}{}{}{}\n"
    measure_format = (note_format * measure_length)

    with open(file, "a") as sm:
        sm.writelines(measure_format.format(*m.flatten()) + ",\n" for m in measures[:-1])
        sm.write(measure_format.format(*measures[-1].flatten()) + ";")


def write_simfile(
    file,
    notes,
    bpm,
    title="Unknown Track",
    artist="Unknown Artist",
    source="Sorce Unavailable",
    offset=-0.000_000,
    rating='hard'
):
    """
    Writes a sim file that can be imported into StepMania
    :param file: File to write to, creating / overwriting it
    :param notes: List-like with shape (num_notes, 4)
    :param bpm: BPM of the track
    :param title: Title of the track
    :param artist: Name of the track's artist
    :param source: Source audio file
    :param offset: Time of the track (in seconds) to start at
    """
    # Make sure the file exists and is empty
    open(file, "w").close()
    write_metadata(file, title, artist, source, offset, bpm)
    write_notes_header(file,rating=rating)
    write_notes(file, notes)
