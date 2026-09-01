import string
import pickle
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# --- Configuration ---
# ⚠️ UPDATE THESE TWO NUMBERS WITH YOUR OUTPUT FROM DAY 3 ⚠️
VOCAB_SIZE = 8766   # Replace with your actual vocabulary size
MAX_LENGTH = 34     # Replace with your actual max sentence length

CAPTIONS_FILE = 'data/captions.txt'
FEATURES_FILE = 'models/features.pkl'
TOKENIZER_FILE = 'models/tokenizer.pkl'
MODEL_SAVE_PATH = 'models/model_weights.keras'

# --- 1. Data Loader & Generator ---
def load_descriptions(filename):
    """Loads and cleans captions directly for the generator"""
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

def data_generator(descriptions, features, tokenizer, max_length, vocab_size, batch_size=32):
    """Yields batches of images and text so we don't crash the RAM."""
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for image_id, desc_list in descriptions.items():
            n += 1
            if image_id not in features:
                continue
            image_feature = features[image_id][0]
            
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(image_feature)
                    X2.append(in_seq)
                    y.append(out_seq)
            
            if n == batch_size:
                yield (np.array(X1), np.array(X2)), np.array(y)
                X1, X2, y = list(), list(), list()
                n = 0

# --- 2. The Model Architecture ---
def define_model(vocab_size, max_length):
    # Feature Extractor (Image Input from ResNet50)
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence Processor (Text Input)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Decoder (Merging both inputs)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    
    # Output Layer (Predicts the next word)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Compile Model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # Print the network structure
    print(model.summary())
    return model

# --- 3. Execution ---
if __name__ == "__main__":
    print("Loading data...")
    # 1. Load image features
    with open(FEATURES_FILE, 'rb') as f:
        features = pickle.load(f)
        
    # 2. Load tokenizer
    with open(TOKENIZER_FILE, 'rb') as f:
        tokenizer = pickle.load(f)
        
    # 3. Load descriptions for generator
    descriptions = load_descriptions(CAPTIONS_FILE)
    
    # 4. Build Model
    print("Building Model...")
    model = define_model(VOCAB_SIZE, MAX_LENGTH)
    
    # 5. Training Setup
    epochs = 10
    batch_size = 32
    steps = len(descriptions) // batch_size
    
    # Save the best model automatically
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='loss', verbose=1, save_best_only=True, mode='min')
    
    print("\nStarting Training! This will take a while...")
    # Initialize the generator
    generator = data_generator(descriptions, features, tokenizer, MAX_LENGTH, VOCAB_SIZE, batch_size)
    
    # Train
    model.fit(
        generator, 
        epochs=epochs, 
        steps_per_epoch=steps, 
        verbose=1,
        callbacks=[checkpoint]
    )
    print("Training Complete! The model weights are saved.")