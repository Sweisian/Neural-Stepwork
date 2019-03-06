import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

def load_training_data():
    #TODO: x_train should be a 2D 2000 by n array where n is number of training examples
    #TODO: y_train should be a 3D 2000 by 4 by n array where n is number of training examples
    pass

def train_model():
    (x_train,y_train) = load_training_data()
    #notes in a stepchart have sequential dependence
    model = Sequential()
    model.add(LSTM(256,input_shape=(x_train.shape[0],1))) #2000 rows by 1 col (1 if note there, 0 if not)
    model.add(Dropout(0.2))
    model.add(Dense((y_train.shape[0],y_train.shape[1]), activation='softmax')) #2000 rows by 4 cols
    model.compile(loss='categorical_crossentropy', optimizer='adam') #TODO: not sure about this line ????
    model.fit(x_train, y_train, epochs=20, batch_size=1) # , callbacks=callbacks_list)
    #TODO: save model

def predict_with_model(model,notes,int_to_line):
    # pick a random seed
    start = numpy.random.randint(0, len(notes) - 1)
    pattern = notes[start]
    for i in range(2000):
        prediction = model.predict(notes, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_line[index]
        seq_in = [int_to_line[value] for value in pattern]
        pattern.append(index)
        pattern = pattern[1:len(pattern)]