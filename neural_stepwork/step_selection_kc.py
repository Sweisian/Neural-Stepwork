import numpy as np
import os
import json
import csv
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers import LSTM, Embedding
import re

possible_lines = {'0000': 0, '0001': 1, '0002': 2, '0010': 3, '0011': 4, '0012': 5, '0020': 6, '0021': 7, '0022': 8, '0100': 9, '0101': 10, '0102': 11, '0110': 12, '0111': 13, '0112': 14, '0120': 15, '0121': 16, '0122': 17, '0200': 18, '0201': 19, '0202': 20, '0210': 21, '0211': 22, '0212': 23, '0220': 24, '0221': 25, '0222': 26, '1000': 27, '1001': 28, '1002': 29, '1010': 30, '1011': 31, '1012': 32, '1020': 33, '1021': 34, '1022': 35, '1100': 36, '1101': 37, '1102': 38, '1110': 39, '1111': 40, '1112': 41, '1120': 42, '1121': 43, '1122': 44, '1200': 45, '1201': 46, '1202': 47, '1210': 48, '1211': 49, '1212': 50, '1220': 51, '1221': 52, '1222': 53, '2000': 54, '2001': 55, '2002': 56, '2010': 57, '2011': 58, '2012': 59, '2020': 60, '2021': 61, '2022': 62, '2100': 63, '2101': 64, '2102': 65, '2110': 66, '2111': 67, '2112': 68, '2120': 69, '2121': 70, '2122': 71, '2200': 72, '2201': 73, '2202': 74, '2210': 75, '2211': 76, '2212': 77, '2220': 78, '2221': 79, '2222': 80}


#{0: '0000', 1: '0001', 2: '0002', 3: '0010', 4: '0011', 5: '0012', 6: '0020', 7: '0021', 8: '0022', 9: '0100', 10: '0101', 11: '0102', 12: '0110', 13: '0111', 14: '0112', 15: '0120', 16: '0121', 17: '0122', 18: '0200', 19: '0201', 20: '0202', 21: '0210', 22: '0211', 23: '0212', 24: '0220', 25: '0221', 26: '0222', 27: '1000', 28: '1001', 29: '1002', 30: '1010', 31: '1011', 32: '1012', 33: '1020', 34: '1021', 35: '1022', 36: '1100', 37: '1101', 38: '1102', 39: '1110', 40: '1111', 41: '1112', 42: '1120', 43: '1121', 44: '1122', 45: '1200', 46: '1201', 47: '1202', 48: '1210', 49: '1211', 50: '1212', 51: '1220', 52: '1221', 53: '1222', 54: '2000', 55: '2001', 56: '2002', 57: '2010', 58: '2011', 59: '2012', 60: '2020', 61: '2021', 62: '2022', 63: '2100', 64: '2101', 65: '2102', 66: '2110', 67: '2111', 68: '2112', 69: '2120', 70: '2121', 71: '2122', 72: '2200', 73: '2201', 74: '2202', 75: '2210', 76: '2211', 77: '2212', 78: '2220', 79: '2221', 80: '2222'}
#
# for i in range(3):
#     for j in range(3):
#         for k in range(3):
#             for l in range(3):
#                 line = str(i) + str(j) + str (k) + str(l)
#                 num = int(line, 3)
#                 possible_lines.update({line : num})
# print(possible_lines)


def load_training_data():
    """
    x_train should be a 2D 2000 by n array where n is number of training examples
    y_train should be a 3D 2000 by 4 by n array where n is number of training examples
    :return: x_train, y_train
    """
    DATA_DIR = "../training/json"
    y_train = list()
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(DATA_DIR, file)) as f:
            step_file = json.load(f)
            print(step_file['title'])
        for track in step_file["notes"]:
            offset = 0
            note_offsets = []
            chart = track['notes']
            for line in chart:
                print(line)
            return
            for line in chart:
                if line != [0, 0, 0, 0]:
                    line=line[:4]
                    note_offsets.append(offset)
                    #print(line)
                    str_line = ''.join(str(s) for s in line)
                    y = possible_lines[str_line]
                    print(step_file['title'],line,str_line,y)
                    y_train.append(y)
                offset += 1
            #print(step_file['title'], y_train[:32])


#x_train, y_train = load_training_data()
# print(x_train[:40])
# print("\n\n")
# print(y_train[:40])

def extract_chart_from_simfile():
    DATA_DIR = "../training/raw"
    output = open('training.csv', 'w')
    writer = csv.writer(output)
    for songFolders in os.scandir(DATA_DIR):
        for entry in os.scandir(songFolders.path):
            name = (os.path.splitext(entry.name))
            print(name)
            extention = name[1].lower()
            if extention !='.sm':
                continue
            with open(entry.path, 'r') as simfile:
                chart = simfile.read()
                chart = chart.split("#NOTES:")
                chart = chart[1]
                difficulty = chart.find("Hard")
                if difficulty == -1:
                    difficulty = chart.find("Challenge")
                if difficulty == -1:
                    difficulty = chart.find("Medium")
                if difficulty == -1:
                    continue
                chart=chart[difficulty:]
                chart = chart[chart.find("0000"):]
                chart=chart[:chart.find(";")]
                chart1D = []
                chart = chart.split("\n")
                for line in chart:
                    if len(line)!= 4:
                        continue
                    line = line.replace('3','2')
                    line = line.replace('M','0')
                    line = line.replace('4','2')
                    chart1D.append(possible_lines[line])
                print(chart1D)
                writer.writerow(chart1D)
                break
    output.close()

extract_chart_from_simfile()
