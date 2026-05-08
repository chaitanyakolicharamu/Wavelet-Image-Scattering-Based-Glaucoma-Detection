import os

base_path = r"C:\Users\venka\Downloads\RIM-ONE_DL_images\RIM-ONE_DL_images"

for root, dirs, files in os.walk(base_path):
    print(root)
    print("Number of files:", len(files))
    print("-"*40)