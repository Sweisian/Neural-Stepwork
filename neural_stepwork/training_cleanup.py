#remove all files except simfile and .wav file
#convert ogg audio to .wav
#remove folders with mp3 audio (harder to convert to wav)

import soundfile as sf
import shutil
import os
import csv

directory = os.fsencode("training")


def file_folder_cleanup():
    extentionsToRemove = ['.avi','.png','.dwi','.old','.ssc','.db','.lrc']
    audioExt = ['.ogg']
    for songFolders in os.scandir(directory):

        for entry in os.scandir(songFolders.path):
            if entry.is_dir():
                shutil.rmtree(entry.path)
            elif entry.is_file():
                name = (os.path.splitext(entry.name))
                extention = name[1].decode("utf-8").lower()
                if extention == '.mp3':
                    shutil.rmtree(songFolders.path.decode("utf-8"))
                    continue
                name = str(name[0])
                if extention in extentionsToRemove:
                    try:
                        os.remove(entry.path)
                        continue
                    except:
                        continue
                if extention in audioExt:
                    data, samplerate = sf.read(entry.path.decode("utf-8"))
                    title = entry.name.decode("utf-8").split(".")[0]
                    sf.write(songFolders.path.decode("utf-8")+'/'+title+'.wav', data, samplerate)
    for songFolders in os.scandir(directory):
        foundSM = False
        foundWav = False
        for entry in os.scandir(songFolders.path):
            name = (os.path.splitext(entry.name))
            extention = name[1].decode("utf-8").lower()
            if extention =='.sm':
                foundSM=True
            if extention == '.wav':
                foundWav = True
            if (extention != '.wav') and (extention != '.sm'):
                print("remove ",entry.path)
                os.remove(entry.path)
        if not foundWav and not foundSM== False: #need simfile for testing/training
            print("removing folder, need simfile and audio", "|||",songFolders.path)
            shutil.rmtree(songFolders.path.decode("utf-8"))

def extract_chart_from_simfile():
    output = open('training.csv', 'w')
    writer = csv.writer(output)
    for songFolders in os.scandir(directory):
        for entry in os.scandir(songFolders.path):
            name = (os.path.splitext(entry.name))
            print(name)
            extention = name[1].decode("utf-8").lower()
            if extention !='.sm':
                continue
            with open(entry.path, 'r') as simfile:
                chart = simfile.read()
                chart = chart.split("#NOTES:")
                chart = chart[1]
                chart = chart[chart.find("0000"):]
                chart1D = ""
                chart = chart.split("\n")
                for line in chart:
                    if line.find("measure") != -1:
                        continue
                    if len(line)!= 4:
                        continue
                    line = line.replace('2','1')
                    line = line.replace('3','1')
                    line = line.replace('M','0')
                    notes = ""
                    for note in line:
                        if note == '0':
                            notes += "0"
                        else:
                            notes+="1"
                    chart1D += notes
                chart1D = chart1D[:-1]
                writer.writerow(chart1D)
                break
    output.close()


#extract_chart_from_simfile()