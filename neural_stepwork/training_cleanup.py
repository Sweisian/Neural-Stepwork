#remove all files except simfile and .wav file
#convert ogg audio to .wav
#remove folders with mp3 audio (harder to convert to wav)

import os
import soundfile as sf
import shutil
import os


def file_folder_cleanup():
    directory = os.fsencode("training")
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
    