from keras.applications.vgg16 import VGG16 
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from visualization import plot_on_grid

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

# model analysis
test_set = testing_data_generator.flow_from_directory('Dataset/PetImages/Test/',
                                                      target_size=(INPUT_SIZE, INPUT_SIZE),
                                                      batch_size=1,
                                                      class_mode='binary')

strongly_wrong_idx = []
strongly_right_idx = []
weakly_wrong_idx = []

for i in range(test_set.__len__()):
    img = test_set.__getitem__(i)[0]
    pred_prob = model.predict(img)[0][0]
    pred_label = int(pred_prob > 0.5)
    actual_label = int(test_set.__getitem__(i)[1][0])
    
    if pred_label != actual_label and (pred_prob > 0.8 or pred_prob < 0.2):
        strongly_wrong_idx.append(i)
    elif pred_label != actual_label and (pred_prob > 0.4 and pred_prob < 0.6):
        weakly_wrong_idx.append(i)
    elif pred_label == actual_label and (pred_prob > 0.8 or pred_prob < 0.2):
        strongly_right_idx.append(i)
        
    if (len(strongly_right_idx) >= 9 and len(strongly_wrong_idx) >= 9 and len(weakly_wrong_idx) >= 9):
        break    

plot_on_grid(test_set, strongly_right_idx)
plot_on_grid(test_set, strongly_wrong_idx)
plot_on_grid(test_set, weakly_wrong_idx)