import os
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Configuration ---
DIRECTORY = 'data/images/Images'
FEATURES_FILE = 'models/features.pkl'

def extract_features(directory):
    """
    Loads ResNet50, processes every image in the directory, 
    and returns a dictionary of image features.
    """
    # 1. Load the ResNet50 model
    # include_top=False removes the final classification layer so we get raw features
    print("Loading ResNet50 model (this might take a few seconds)...")
    model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    
    features = dict()
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Could not find the folder '{directory}'.")
        return None
        
    images = os.listdir(directory)
    total_images = len(images)
    print(f"Found {total_images} files. Starting extraction...")
    
    # 2. Loop through every image
    for i, img_name in enumerate(images):
        # Skip anything that isn't a jpg
        if not img_name.endswith('.jpg'):
            continue
            
        img_path = os.path.join(directory, img_name)
        
        try:
            # ResNet50 strictly requires images to be exactly 224x224 pixels
            image = load_img(img_path, target_size=(224, 224))
            
            # Convert pixels to a numpy array
            image = img_to_array(image)
            
            # Reshape data for the model (1 image, 224 height, 224 width, 3 color channels)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            
            # Normalize pixel values specifically for ResNet50
            image = preprocess_input(image)
            
            # Pass through the CNN to get the math vector
            feature = model.predict(image, verbose=0)
            
            # Get the image ID (remove '.jpg' from the string)
            image_id = img_name.split('.')[0]
            
            # Store feature in our dictionary
            features[image_id] = feature
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            
        # Print progress every 500 images so you know it hasn't frozen
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1} / {total_images} images...")
            
    return features

# --- Execution ---
if __name__ == "__main__":
    # Create a 'models' folder to save our output if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Run the extraction function
    features = extract_features(DIRECTORY)
    
    if features:
        print(f"\nExtraction complete! Saving {len(features)} features to disk...")
        # Save the dictionary as a binary file
        with open(FEATURES_FILE, 'wb') as f:
            pickle.dump(features, f)
        print(f"Success! Features saved to {FEATURES_FILE}")