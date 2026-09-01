import string
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# --- Configuration ---
CAPTIONS_FILE = 'data/captions.txt'
TOKENIZER_FILE = 'models/tokenizer.pkl'

def get_cleaned_descriptions():
    """Reads the text file and returns a dictionary of clean captions."""
    with open(CAPTIONS_FILE, 'r', encoding='utf-8') as file:
        doc = file.read()
        
    mapping = dict()
    # Skip the first line if it's the 'image,caption' header
    lines = doc.split('\n')
    if "image,caption" in lines[0].lower():
        lines = lines[1:]
        
    for line in lines:
        if len(line) < 2: continue
        tokens = line.split(',')
        if len(tokens) < 2: continue
        
        image_id = tokens[0].split('.')[0]
        desc = ' '.join(tokens[1:])
        
        # Clean text: lowercase, remove punctuation, remove numbers/single letters
        desc = desc.split()
        desc = [word.lower() for word in desc]
        desc = [word.translate(str.maketrans('', '', string.punctuation)) for word in desc]
        desc = [word for word in desc if len(word) > 1 and word.isalpha()]
        
        # Add start and end tokens so the model knows when a sentence begins/ends
        desc = 'startseq ' + ' '.join(desc) + ' endseq'
        
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(desc)
    return mapping

def create_tokenizer(descriptions):
    """Creates the vocabulary mapping words to numbers."""
    all_captions = []
    for key in descriptions.keys():
        [all_captions.append(cap) for cap in descriptions[key]]
        
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer, all_captions

def max_length(captions):
    """Finds the longest caption so we can pad all sentences to the same length."""
    return max(len(cap.split()) for cap in captions)

# --- Execution ---
if __name__ == "__main__":
    print("1. Loading and cleaning text...")
    descriptions = get_cleaned_descriptions()
    
    print("2. Building the vocabulary...")
    tokenizer, all_captions = create_tokenizer(descriptions)
    
    # We add +1 to the vocab size to account for the "0" padding token
    vocab_size = len(tokenizer.word_index) + 1 
    print(f"-> Vocabulary Size: {vocab_size} unique words")
    
    max_len = max_length(all_captions)
    print(f"-> Maximum sentence length: {max_len} words")
    
    print("3. Saving Tokenizer to disk...")
    with open(TOKENIZER_FILE, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print(f"Success! Saved tokenizer to {TOKENIZER_FILE}")