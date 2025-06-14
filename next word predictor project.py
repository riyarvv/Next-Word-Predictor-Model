import numpy as np
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq

path = r"C:\Users\Riya R V\Downloads\1661-0.txt"
text = open(path, encoding="utf-8").read().lower()

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

WORD_LENGTH = 5
prev_words = []
next_words = []
for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i + WORD_LENGTH])

X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=np.float32)
Y = np.zeros((len(next_words), len(unique_words)), dtype=np.float32)
for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[next_words[i]]] = 1
    
model = Sequential()
model.add(LSTM(128, input_shape=(WORD_LENGTH, len(unique_words))))
model.add(Dense(len(unique_words)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True).history

model.save('keras_next_word_model.h5')
pickle.dump(history, open("history.p", "wb"))
model = load_model('keras_next_word_model.h5')
history = pickle.load(open("history.p", "rb"))

plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

def prepare_input(text):
    words_list = text.lower().split()
    x = np.zeros((1, WORD_LENGTH, len(unique_words)), dtype=np.float32)
    for t, word in enumerate(words_list):
        if word in unique_word_index:
            x[0, t, unique_word_index[word]] = 1
    return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10)  # prevent log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

# Build reverse index
index_word = dict((i, c) for c, i in unique_word_index.items())

def predict_next_words(seed_text, n=3):
    tokenized = seed_text.lower().split()
    if len(tokenized) < WORD_LENGTH:
        raise ValueError(f"Input must be at least {WORD_LENGTH} words long.")
    tokenized = tokenized[-WORD_LENGTH:]  # take last WORD_LENGTH words
    x = prepare_input(' '.join(tokenized))
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [index_word[idx] for idx in next_indices]

# Sample quotes
quotes = [
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
    "That which does not kill us makes us stronger.",
    "I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
    "And those who were seen dancing were thought to be insane by those who could not hear the music.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!"
]

for q in quotes:
    seed = ' '.join(q.lower().split()[:WORD_LENGTH])  # first 5 words
    print(f"Input: {seed}")
    predictions = predict_next_words(seed, 5)
    print("Predicted next words:", predictions)
    