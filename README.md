# Image Caption Generator (CNN-LSTM)

An end-to-end Machine Learning pipeline that generates descriptive English captions for uploaded images. This project utilizes a Convolutional Neural Network (ResNet50) for visual feature extraction and a Long Short-Term Memory (LSTM) network for sequential text generation.

## 🧠 Architecture
1. **Encoder (CNN):** A pre-trained ResNet50 model extracts a 2048-dimensional feature vector from the input image.
2. **Decoder (RNN/LSTM):** An embedding layer and LSTM network process the text sequences.
3. **Merge Layer:** The visual vector and text sequences are combined to predict the next word in the caption probability distribution.
### 📐 System Flow Diagram
![CNN-LSTM Architecture Diagram](https://raw.githubusercontent.com/Shivam0318/Image-Caption-Generator/main/architecture.png)

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV, ResNet50
* **NLP:** NLTK (BLEU Evaluation), Keras Tokenizer
* **Interface:** Streamlit
* **Language:** Python 3.10

## 🚀 Installation & Local Execution
1. Clone the repository:
   ```bash
   git clone [https://github.com/Shivam0318/Image-Caption-Generator.git](https://github.com/Shivam0318/Image-Caption-Generator.git)

```

2. Install the required dependencies:
```bash
pip install -r requirements.txt

```


3. Download the Flickr8k dataset and place it in the `/data` directory.
4. Run the feature extraction and model training scripts.
5. Launch the Streamlit web interface:
```bash
python -m streamlit run app.py

```



## 📊 Evaluation

The model's accuracy was evaluated against human-annotated captions using the Bilingual Evaluation Understudy (BLEU) metric via the NLTK library. The baseline model achieved the following scores:

* **BLEU-1:** 0.5329
* **BLEU-2:** 0.3412
* **BLEU-3:** 0.2280
* **BLEU-4:** 0.1484

## 👨‍💻 Author

**Shivam Yadav**
