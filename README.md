# Next Word Prediction using LSTM

This project is a deep learning model built using TensorFlow/Keras to predict the next word in a sequence. It uses an LSTM (Long Short-Term Memory) neural network trained on a text corpus to generate predictions based on previous words.

---

## 📚 Project Description

The model takes in a sequence of **5 words** and predicts the **next most probable word** from the training corpus. It uses a recurrent neural network (RNN) with LSTM layers to learn sequential patterns in text.

---

## 📁 Dataset

The model is trained on text from Project Gutenberg's The Adventures of Sherlock Holmes, by Arthur Conan Doyle.

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
   git clone https://github.com/your-username/next-word-predictor.git
   cd next-word-predictor

2. Install Dependencies:
   pip install numpy nltk matplotlib tensorflow

3. Download the text file given below and update the path:
   [File](https://drive.google.com/file/d/1GeUzNVqiixXHnTl8oNiQ2W3CynX_lsu2/view?usp=sharing)
   path = r"C:\Users\YourName\Downloads\1661-0.txt"

---

▶️ How to Run
1. Train the model:
   python next_word_predictor.py

   
2. After training, the model and training history will be saved:
   keras_next_word_model.h5
   history.p

3. Predict next words using:
   predict_next_words("your 5 word input here", 5)

---

📊 Output
The script also plots training and validation accuracy/loss:
   Accuracy over epochs
   Loss over epochs

---

💡 Sample Predictions
  ![Screenshot 2025-06-14 203424](https://github.com/user-attachments/assets/b4dab4f7-b456-4eee-ac2a-d98df688b278)




