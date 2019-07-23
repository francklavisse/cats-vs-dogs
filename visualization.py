from matplotlib import pyplot as plt 
import os 
import random 

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

_, _, cat_images = next(os.walk('Dataset/PetImages/Cat'))

fig, ax = plt.subplots(3,3, figsize=(20,10))

for idx, img in enumerate(random.sample(cat_images, 9)):
    img_read = plt.imread('Dataset/PetImages/Cat/' + img)
    ax[int(idx / 3), idx % 3].imshow(img_read)
    ax[int(idx / 3), idx % 3].axis('off')
    ax[int(idx / 3), idx % 3].set_title('Cat/' + img)

plt.show()