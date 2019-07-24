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

def plot_on_grid(test_set, idx_to_plot, img_size=128):
    fig, ax = plt.subplots(3, 3, figsize=(20,10))
    for i, idx in enumerate(random.sample(idx_to_plot, 9)):
        img = test_set.__getitem__(idx)[0].reshape(img_size, img_size, 3)
        ax[int(i / 3), i % 3].imshow(img)
        ax[int(i / 3), i % 3].axis('off')    
        
    plt.show()
    
display_random_pictures('Dog')