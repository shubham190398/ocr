import os
from PIL import Image, ImageSequence

folder = "dataset/Images-B"
count = 1

for file_name in os.listdir(folder):
    img = Image.open(f"dataset/Images-B/{file_name}")

    for i, im in enumerate(ImageSequence.Iterator(img)):
        if i == 1:
            im.save(f"dataset/consolidated_remittances/{count}_B.png")

    count += 1
