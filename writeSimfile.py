import os

def writeSimfile(title="",artist="",music="",offset=-0.000000,bpms=128):
    songMetadata = "#TITLE:{0};\n#ARTIST:{1};\n#MUSIC:{2};\n#OFFSET:{3};\n#BPMS:0.0={4};\n#STOPS:;\n".format(title,artist,music,offset,bpms)
    # setting difficulty level to 1
    # setting difficulty mode to edit (instead of hard, expert, etc.)
    # setting play mode to dance-singles
    # the series of zeros represents groove radar values; these values aren't required for now 
    chartMetadata ="#NOTES:\n\tdance-single:\n\t:\n\tEdit:\n\t1:\n\n0.0,0.0,0.0,0.0,0.0:\n"
    measures = ""
    for i in range(30): #for now im saying there are 30 measures in the song
        measures += (generateRandomMeasure() + ",\n") #measures separated by commas
    measures=measures[:-2] #remove last comma and new line character
    simfileContent = songMetadata + chartMetadata + measures
    simfileContentLines = simfileContent.split("\n")
    os.makedirs('/'+title)
    file = open(title+'.sm', 'w')
    #TODO put audio file in folder too
    for line in simfileContentLines:
        file.write(line+"\n")
    file.close()

def generateRandomMeasure():
    return "1000\n0100\n1000\n0010\n0100\n0001\n0010\n0100\n"

writeSimfile("test","test","test.wav")
