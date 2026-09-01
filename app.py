import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration ---
MAX_LENGTH = 34
TOKENIZER_FILE = 'models/tokenizer.pkl'
MODEL_FILE = 'models/model_weights.keras'

# --- 1. Caching Models to Memory ---
@st.cache_resource
def load_assets():
    with open(TOKENIZER_FILE, 'rb') as f:
        tokenizer = pickle.load(f)
    model = load_model(MODEL_FILE)
    resnet_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    return tokenizer, model, resnet_model

tokenizer, model, resnet_model = load_assets()
word_for_id = {index: word for word, index in tokenizer.word_index.items()}

# --- 2. Processing Functions ---
def extract_features(image):
    # Convert Streamlit uploaded image to exactly what ResNet50 expects
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = resnet_model.predict(image, verbose=0)
    return feature

def generate_caption(feature):
    in_text = 'startseq'
    for _ in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id.get(yhat)
        
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
            
    return in_text.replace('startseq ', '').replace(' endseq', '')

# --- 3. Streamlit Web UI ---
st.title("📷 Image Caption Generator")
st.write("Upload any image and the CNN-LSTM model will describe it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image on the web page
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Generate button
    if st.button("Generate Caption"):
        with st.spinner("Analyzing image..."):
            feature = extract_features(image)
            caption = generate_caption(feature)
            
            st.success("Generation Complete!")
            st.markdown(f"### 🤖 Prediction: *{caption.capitalize()}*")