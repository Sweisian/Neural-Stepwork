import os
import time
import argparse
from shutil import copy

from neural_stepwork.simfile import write_simfile
from neural_stepwork.find_bpm import get_bpm

PREFIX = "Run_"
OUTPUT_DIR = "output/"
EXTENSION = ".wav"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create StepMania simfile.")
    parser.add_argument(
        "-f", "--file", required=True, dest="file", help="file or directory to parse"
    )

    args = parser.parse_args()

    files = list()

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

    dest = OUTPUT_DIR + PREFIX + str(int(time.time())) + "/"

    if not os.path.isdir(OUTPUT_DIR + dest):
        os.mkdir(dest)

    for source_file in files:
        copy(source_file, dest)
        name = os.path.basename(source_file).split(".wav")[0]
        print("Processing {}".format(name))

        write_simfile(source_file, dest, name, get_bpm(source_file))
