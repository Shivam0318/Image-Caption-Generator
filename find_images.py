import os

print("Searching for the images...")
for root, dirs, files in os.walk('.'):
    # Count how many .jpg files are in the current folder
    jpgs = [f for f in files if f.endswith('.jpg')]
    
    if len(jpgs) > 1000:
        print(f"\nFOUND THEM!")
        print(f"They are hiding in this folder: {root}")
        print(f"\nGo to 2_feature_extractor.py and change Line 7 exactly to this:")
        print(f"DIRECTORY = r'{root}'")
        break