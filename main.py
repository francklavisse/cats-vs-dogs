from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

FILTER_SIZE = 3
NUM_FILTERS = 32
INPUT_SIZE = 32
MAXPOOL_SIZE = 2
BATCH_SIZE = 16
STEPS_PER_EPOCH = 20000 / BATCH_SIZE 
EPOCHS = 10

def ConvolutionPooling(model):
    model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE),
                    input_shape = (INPUT_SIZE, INPUT_SIZE, 3),
                    activation = 'relu'))

    model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))
    return model

model = ConvolutionPooling(ConvolutionPooling(model))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])