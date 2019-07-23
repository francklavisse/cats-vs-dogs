import os 
import random 
import shutil 
import piexif

def remove_exif_data(src_folder):
    _, _, images = next(os.walk(src_folder))
    for img in images:
        try:
            piexif.remove(src_folder + img)
        except:
            pass

def  train_test_split(src_folder, train_size = 0.8):
    # remove old dirs
    shutil.rmtree(src_folder + 'Train/Cat/', ignore_errors=True)
    shutil.rmtree(src_folder + 'Train/Dog/', ignore_errors=True)
    shutil.rmtree(src_folder + 'Test/Cat/', ignore_errors=True)
    shutil.rmtree(src_folder + 'Test/Dog/', ignore_errors=True)

    # create new dirs
    os.makedirs(src_folder + 'Train/Cat/')
    os.makedirs(src_folder + 'Train/Dog/')
    os.makedirs(src_folder + 'Test/Cat/')
    os.makedirs(src_folder + 'Test/Dog/')

    _, _, cat_images = next(os.walk(src_folder + 'Cat/'))
    files_to_be_removed = ['Thumbs.db', '666.jpg', '835.jpg']
    for file in files_to_be_removed:
        cat_images.remove(file)
    num_cat_images = len(cat_images)
    num_cat_images_train = int(train_size * num_cat_images)
    num_cat_images_test = num_cat_images - num_cat_images_train

    _, _, dog_images = next(os.walk(src_folder + 'Dog/'))
    files_to_be_removed = ['Thumbs.db', '11702.jpg']
    for file in files_to_be_removed:
        dog_images.remove(file)
    num_dog_images = len(dog_images)
    num_dog_images_train = int(train_size * num_dog_images)
    num_dog_images_test = num_dog_images - num_dog_images_train

    cat_train_images = random.sample(cat_images, num_cat_images_train)
    for img in cat_train_images:
        shutil.copy(src=src_folder + 'Cat/' + img, dst=src_folder + 'Train/Cat/')
    cat_test_images = [img for img in cat_images if img not in cat_train_images]
    for img in cat_test_images:
        shutil.copy(src=src_folder + 'Cat/' + img, dst=src_folder + 'Test/Cat/')

    dog_train_images = random.sample(dog_images, num_dog_images_train)
    for img in dog_train_images:
        shutil.copy(src=src_folder + 'Dog/' + img, dst=src_folder + 'Train/Dog/')
    dog_test_images = [img for img in dog_images if img not in dog_train_images]
    for img in dog_test_images:
        shutil.copy(src=src_folder + 'Dog/' + img, dst=src_folder + 'Test/Dog/')

    remove_exif_data(src_folder + 'Train/Dog/')
    remove_exif_data(src_folder + 'Test/Dog/')
    remove_exif_data(src_folder + 'Train/Cat/')
    remove_exif_data(src_folder + 'Test/Cat/')

src_folder = 'Dataset/PetImages/'
train_test_split(src_folder)