# Next Word Prediction using LSTM

This project is a deep learning model built using TensorFlow/Keras to predict the next word in a sequence. It uses an LSTM (Long Short-Term Memory) neural network trained on a text corpus to generate predictions based on previous words.


## 📚 Project Description

The model takes in a sequence of **5 words** and predicts the **next most probable word** from the training corpus. It uses a recurrent neural network (RNN) with LSTM layers to learn sequential patterns in text.

---

## 📁 Dataset

The model is trained on text from [Project Gutenberg](https://www.gutenberg.org/) — specifically, a `.txt` file such as `1661-0.txt`.

> 🔤 The text is lowercased and tokenized using regular expressions to extract words only.

---

## 🏗️ Model Architecture

- **Input Shape:** (5 words, one-hot encoded vector)
- **Layers:**
  - `LSTM(128)` layer
  - `Dense` layer with `softmax` activation
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** RMSprop
- **Training Samples:** 5,000 (can be increased)
- **Epochs:** 2 (for demo, can be increased)

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/next-word-predictor.git
   cd next-word-predictor

