import string
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu

# --- Configuration ---
MAX_LENGTH = 34
CAPTIONS_FILE = 'data/captions.txt'
FEATURES_FILE = 'models/features.pkl'
TOKENIZER_FILE = 'models/tokenizer.pkl'
MODEL_FILE = 'models/model_weights.keras'
TEST_SAMPLE_SIZE = 100  # Number of images to test to save CPU time

# --- 1. Load Data & Models ---
def load_descriptions(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        doc = file.read()
    
    mapping = dict()
    lines = doc.split('\n')
    if "image,caption" in lines[0].lower():
        lines = lines[1:]
        
    for line in lines:
        if len(line) < 2: continue
        tokens = line.split(',')
        image_id = tokens[0].split('.')[0]
        desc = ' '.join(tokens[1:])
        
        desc = desc.split()
        desc = [word.lower() for word in desc]
        desc = [word.translate(str.maketrans('', '', string.punctuation)) for word in desc]
        desc = [word for word in desc if len(word) > 1 and word.isalpha()]
        desc = 'startseq ' + ' '.join(desc) + ' endseq'
        
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(desc)
    return mapping

print("Loading data and model...")
descriptions = load_descriptions(CAPTIONS_FILE)

with open(FEATURES_FILE, 'rb') as f:
    features = pickle.load(f)

with open(TOKENIZER_FILE, 'rb') as f:
    tokenizer = pickle.load(f)

word_for_id = {index: word for word, index in tokenizer.word_index.items()}
model = load_model(MODEL_FILE)

# --- 2. Generation Function ---
def generate_caption(model, tokenizer, max_length, feature):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq ', '').replace(' endseq', '')

# --- 3. Evaluation ---
print(f"Starting evaluation on {TEST_SAMPLE_SIZE} images...")
actual, predicted = list(), list()
count = 0

for image_id, caption_list in descriptions.items():
    if count >= TEST_SAMPLE_SIZE:
        break
    if image_id not in features:
        continue
        
    # Generate a prediction for this image
    feature = features[image_id][0].reshape((1, 2048))
    yhat = generate_caption(model, tokenizer, MAX_LENGTH, feature)
    
    # Store actual captions (cleaned of startseq/endseq)
    ground_truths = [cap.replace('startseq ', '').replace(' endseq', '').split() for cap in caption_list]
    actual.append(ground_truths)
    
    # Store predicted caption
    predicted.append(yhat.split())
    
    count += 1
    if count % 10 == 0:
        print(f"Evaluated {count}/{TEST_SAMPLE_SIZE} images...")

# --- 4. Calculate BLEU Scores ---
print("\n=== BLEU SCORES ===")
# BLEU-1: Matches individual words
print(f"BLEU-1: {corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)):.4f}")
# BLEU-2: Matches two-word phrases
print(f"BLEU-2: {corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)):.4f}")
# BLEU-3: Matches three-word phrases
print(f"BLEU-3: {corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0)):.4f}")
# BLEU-4: Matches four-word phrases
print(f"BLEU-4: {corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)):.4f}")