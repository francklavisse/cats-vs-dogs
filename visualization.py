from matplotlib import pyplot as plt 
import os 
import random 

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def display_random_pictures(prefix='Cat'):
    _, _, pet_images = next(os.walk('Dataset/PetImages/' + prefix))

    fig, ax = plt.subplots(3,3, figsize=(20,10))

    for idx, img in enumerate(random.sample(pet_images, 9)):
        img_read = plt.imread('Dataset/PetImages/' + prefix + '/' + img)
        ax[int(idx / 3), idx % 3].imshow(img_read)
        ax[int(idx / 3), idx % 3].axis('off')
        ax[int(idx / 3), idx % 3].set_title(prefix + '/' + img)

    plt.show()
    
display_random_pictures('Dog')