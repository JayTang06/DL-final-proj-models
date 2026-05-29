from PIL import Image
import os


directory = os.fsencode('train_images')
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        img = Image.open(f'train_images/{filename}')

        new_image = img.resize((512, 512))

        new_image.save(f'train_images_resized_512/{filename}')
        continue
    else:
        continue

