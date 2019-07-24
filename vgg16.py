from keras.applications.vgg16 import VGG16 
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

INPUT_SIZE = 128

vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(INPUT_SIZE, INPUT_SIZE, 3))

for layer in vgg16.layers:
    layer.trainable = False 
    
input_ = vgg16.input 
output_ = vgg16(input_)
last_layer = Flatten(name='flatten')(output_)
last_layer= Dense(1, activation='sigmoid')(last_layer)
model = Model(input=input_, output=last_layer)

BATCH_SIZE = 16
STEPS_PER_EPOCH = 200
EPOCHS = 3

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

training_data_generator = ImageDataGenerator(rescale = 1./255)
testing_data_generator = ImageDataGenerator(rescale = 1./255)

training_set = training_data_generator.flow_from_directory('Dataset/PetImages/Train/',
                                                           target_size=(INPUT_SIZE, INPUT_SIZE),
                                                           batch_size=BATCH_SIZE,
                                                           class_mode='binary')

test_set = testing_data_generator.flow_from_directory('Dataset/PetImages/Test/',
                                                      target_size=(INPUT_SIZE, INPUT_SIZE),
                                                      batch_size=BATCH_SIZE,
                                                      class_mode='binary')

model.fit_generator(training_set, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1)