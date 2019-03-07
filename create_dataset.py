import re
import os
import sys
import traceback
import json

import logging as log

json.encoder.FLOAT_REPR = lambda f: ("%.6f" % f)


VALID_PULSES = set([4, 8, 16, 32])
PULSE_LENGTH = 32
REQUIRED_ATTRIBUTES = ["title", "bpms", "notes"]
IN_DIR = "./training/raw/"
OUT_DIR = "./training/json/"
EXTENSION = ".sm"

int_parser = lambda x: int(x.strip()) if x.strip() else None
bool_parser = lambda x: True if x.strip() == "YES" else False
str_parser = lambda x: x.strip() if x.strip() else None
float_parser = lambda x: float(x.strip()) if x.strip() else None


def kv_parser(k_parser, v_parser):
    def parser(x):
        if not x:
            return None, None
        k, v = x.split("=")
        return k_parser(k), v_parser(v)

    return parser


def list_parser(x_parser):
    def parser(l):
        l_strip = l.strip()
        if len(l_strip) == 0:
            return []
        else:
            return [x_parser(x) for x in l_strip.split(",")]

    return parser


def bpms_parser(x):
    bpms = list_parser(kv_parser(float_parser, float_parser))(x)

    if len(bpms) == 0:
        raise ValueError("No BPMs found in list")
    if bpms[0][0] != 0.0:
        raise ValueError("First beat in BPM list is {}".format(bpms[0][0]))

    # make sure changes are non-negative, take last for equivalent
    beat_last = -1.0
    bpms_cleaned = []
    for beat, bpm in bpms:
        if beat is None or bpm is None:
            raise ValueError("Empty BPM found")
        if bpm <= 0.0:
            raise ValueError("Non positive BPM found {}".format(bpm))
        if beat == beat_last:
            bpms_cleaned[-1] = (beat, bpm)
            continue
        bpms_cleaned.append((beat, bpm))
        if beat <= beat_last:
            raise ValueError("Descending list of beats in BPM list")
        beat_last = beat
    if len(bpms) != len(bpms_cleaned):
        log.warning(
            "One or more (beat, BPM) pairs begin on the same beat, using last listed"
        )

    return bpms_cleaned


def notes_parser(x):
    pattern = r"([^:]*):" * 5 + r"([^;:]*)"
    notes_split = re.findall(pattern, x)
    if len(notes_split) != 1:
        raise ValueError("Bad formatting of notes section")
    notes_split = notes_split[0]
    if len(notes_split) != 6:
        raise ValueError("Bad formatting within notes section")

    # parse/clean measures
    measures = [measure.splitlines() for measure in notes_split[5].split(",")]
    measures_clean = list()
    for measure in measures:
        measure_clean = list(
            filter(
                lambda pulse: not pulse.strip().startswith("//")
                and len(pulse.strip()) > 0,
                measure,
            )
        )
        measures_clean.append(measure_clean)
    if len(measures_clean) > 0 and len(measures_clean[-1]) == 0:
        measures_clean = measures_clean[:-1]

    # transform measure to list of ints and pad to PULSE_LENGTH
    flat_notes = list()
    for measure in measures_clean:
        if len(measure) not in VALID_PULSES:
            log.warning(
                "Nonstandard subdivision {} detected, skipping".format(len(measure))
            )
            return None
        pad_length = int(PULSE_LENGTH / len(measure)) - 1
        for beat in measure:
            new_beat = list()
            step_list = list(beat.lower())
            for step in step_list:
                if step == "1":
                    new_beat.append(1)
                if step in ("2", "3", "4"):
                    new_beat.append(2)
                else:
                    new_beat.append(0)
                    flat_notes.append(new_beat)
            for _ in range(pad_length):
                flat_notes.append([0, 0, 0, 0])
    return {
        "type": str_parser(notes_split[0]),
        "desc_or_author": str_parser(notes_split[1]),
        "difficulty_coarse": str_parser(notes_split[2]),
        "difficulty_fine": int_parser(notes_split[3]),
        "groove_radar": list_parser(float_parser)(notes_split[4]),
        "notes": flat_notes,
    }


ATTR_NAME_TO_PARSER = {
    "title": str_parser,
    "subtitle": str_parser,
    "artist": str_parser,
    "titletranslit": str_parser,
    "subtitletranslit": str_parser,
    "artisttranslit": str_parser,
    "genre": str_parser,
    "credit": str_parser,
    "banner": str_parser,
    "background": str_parser,
    "lyricspath": str_parser,
    "cdtitle": str_parser,
    "music": str_parser,
    "offset": float_parser,
    "bpms": bpms_parser,
    "stops": list_parser(kv_parser(float_parser, float_parser)),
    "samplestart": float_parser,
    "samplelength": float_parser,
    "displaybpm": str_parser,
    "selectable": bool_parser,
    "bgchanges": str_parser,
    "bgchanges2": str_parser,
    "fgchanges": str_parser,
    "keysounds": str_parser,
    "musiclength": float_parser,
    "musicbytes": int_parser,
    "notes": notes_parser,
}
ATTR_MULTI = ["notes"]


def parse_sm_txt(text):
    attributes = {attribute_name: [] for attribute_name in ATTR_MULTI}

    for attr_name, attr_val in re.findall(r"#([^:]*):([^;]*);", text):
        attr_name = attr_name.lower()

        if attr_name not in ATTR_NAME_TO_PARSER:
            log.warning(
                "Found unexpected attribute {}:{}, ignoring".format(attr_name, attr_val)
            )
            continue

        attr_val_parsed = ATTR_NAME_TO_PARSER[attr_name](attr_val)
        if not attr_val_parsed:
            continue

        if attr_name in attributes:
            if attr_name not in ATTR_MULTI:
                if attr_val_parsed == attributes[attr_name]:
                    continue
                else:
                    raise ValueError(
                        "Attribute {} defined multiple times".format(attr_name)
                    )
            attributes[attr_name].append(attr_val_parsed)
        else:
            attributes[attr_name] = attr_val_parsed

    to_delete = list()
    for attr_name, attr_val in attributes.items():
        if not attr_val:
            to_delete.append(attr_name)

    for attr_name in to_delete:
        del attributes[attr_name]

    return attributes


def parse_sm_file(path):
    with open(path, "r") as f:
        text = f.read()

    # parse file
    try:
        attributes = parse_sm_txt(text)

    except ValueError as e:
        log.error("{} in\n{}".format(e, sm_fp))
        return None

    except Exception as e:
        log.critical("Unhandled parse exception {}".format(traceback.format_exc()))
        raise e

    # check required attributes
    for attr_name in REQUIRED_ATTRIBUTES:
        if attr_name not in attributes:
            log.error("Missing required attribute {}".format(attr_name))
            return None

    return attributes


if __name__ == "__main__":
    avg_difficulty = 0.0
    num_charts = 0
    num_files = 0

    num_old_files = 0
    for old_file in os.listdir(OUT_DIR):
        file_path = os.path.join(OUT_DIR, old_file)
        if os.path.isfile(file_path) and old_file.endswith(".json"):
            os.unlink(file_path)
            num_old_files += 1

    print("Removed {} old json files".format(num_old_files))

    sm_files = list()
    for root, dirs, files in os.walk(IN_DIR):
        for file in files:
            if file.endswith(EXTENSION):
                path = os.path.join(root, file)
                sm_files.append(path)
                print(path)

    for file in sm_files:
        name = os.path.basename(file).split(".")[0]
        out_path = os.path.join(OUT_DIR, name) + ".json"
        attributes = parse_sm_file(file)
        if not attributes:
            continue

        num_files += 1

        for sm_notes in attributes["notes"]:
            avg_difficulty += float(sm_notes["difficulty_fine"])
            num_charts += 1

        with open(out_path, "w") as out:
            try:
                out.write(json.dumps(attributes))
            except UnicodeDecodeError:
                log.error("Unicode error in {}".format(file))
                continue
    print(
        "Parsed {} stepfiles, {} charts, average difficulty {}".format(
            num_files, num_charts, avg_difficulty / (num_charts + 0.00000001)
        )
    )
