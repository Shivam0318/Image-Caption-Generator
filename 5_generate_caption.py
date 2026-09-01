import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration ---
MAX_LENGTH = 34
TOKENIZER_FILE = 'models/tokenizer.pkl'
MODEL_FILE = 'models/model_weights.keras'
IMAGE_PATH = 'sample_image.jpg' # We will point this to a test image

# --- 1. Load the Tokenizer and Models ---
print("Loading tokenizer...")
with open(TOKENIZER_FILE, 'rb') as f:
    tokenizer = pickle.load(f)

# Create a reverse dictionary to map integers back to actual English words
word_for_id = {index: word for word, index in tokenizer.word_index.items()}

print("Loading your trained model...")
model = load_model(MODEL_FILE)

print("Loading ResNet50 for image processing...")
resnet_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')

# --- 2. Feature Extraction for a Single Image ---
def extract_features(filename):
    """Processes a single new image through ResNet50."""
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = resnet_model.predict(image, verbose=0)
    return feature

# --- 3. Caption Generation ---
def generate_caption(model, tokenizer, max_length, feature):
    """Predicts the caption word by word."""
    # Seed the generation process with the start token
    in_text = 'startseq'
    
    for _ in range(max_length):
        # Convert the current text sequence into integers
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad it to exactly 34 words
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict the probability of every word in the vocabulary being the next word
        yhat = model.predict([feature, sequence], verbose=0)
        # Grab the integer ID of the word with the highest probability
        yhat = np.argmax(yhat)
        
        # Look up the actual word corresponding to that integer
        word = word_for_id.get(yhat)
        
        # Stop if we predict an unknown word
        if word is None:
            break
            
        # Append the predicted word to our text sequence
        in_text += ' ' + word
        
        # Stop if the model predicts the end of the sentence
        if word == 'endseq':
            break
            
    # Clean up the final output for the user
    final_caption = in_text.replace('startseq ', '').replace(' endseq', '')
    return final_caption

# --- Execution ---
if __name__ == "__main__":
    print(f"\nProcessing {IMAGE_PATH}...")
    try:
        photo_feature = extract_features(IMAGE_PATH)
        
        print("Generating caption...")
        caption = generate_caption(model, tokenizer, MAX_LENGTH, photo_feature)
        
        print("\n" + "="*50)
        print(f"RESULT: {caption}")
        print("="*50 + "\n")
    except FileNotFoundError:
        print(f"\nError: Could not find '{IMAGE_PATH}'.")
        print("Please download a random test image, name it 'sample_image.jpg', and put it in this folder!")