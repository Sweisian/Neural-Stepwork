import os
import time
import argparse
from shutil import copy

from neural_stepwork.simfile import write_simfile
from neural_stepwork.find_bpm import get_bpm
from neural_stepwork.note_predictor import generate

PREFIX = "Run_"
OUTPUT_DIR = "output/"
EXTENSION = ".wav"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create StepMania simfile.")
    parser.add_argument("-f", "--file", required=True, dest="file", help="file or directory to parse")
    parser.add_argument("-rating",required=True,dest="rating",help="Difficulty Rating of step chart: easy medium or hard")
    args = parser.parse_args()

    files = list()
    rating = args.rating

    if os.path.isfile(args.file):
        files.append(args.file)

    elif os.path.isdir(args.file):
        for dirpath, _, filenames in os.walk(args.file):
            for f in filenames:
                if not f.endswith(EXTENSION):
                    continue
                files.append(os.path.abspath(os.path.join(dirpath, f)))

    else:
        raise ValueError("Unknown file {}".format(args.file))

    #dest = OUTPUT_DIR + PREFIX + str(int(time.time())) + "/"



    for source_file in files:
        name = os.path.basename(source_file).split(".wav")[0]
        dest = OUTPUT_DIR + name + "_" + str(int(time.time())) + "/"
        if not os.path.isdir(dest):
            os.mkdir(dest)
        copy(source_file, dest)
        print("Processing {}".format(name))
        bpm = get_bpm(source_file)
        notes = generate(source_file,bpm,rating)
        write_simfile(file = (dest + name + ".sm"),notes=notes,bpm=bpm,rating=rating,title=name,source=(name + EXTENSION))
