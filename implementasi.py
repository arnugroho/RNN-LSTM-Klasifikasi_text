# !pip install nltk --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Download the 'punkt' resource along with 'stopwords' and 'punkt_tab'
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense

df = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')
df = df[['label', 'tweet']].rename(columns={'label': 'sentiment', 'tweet': 'text'})
df['sentiment'] = df['sentiment'].map({0: 'negative', 1: 'positive'})
df = df.dropna().reset_index(drop=True)
df.head()

stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(preprocess)

MAX_NUM_WORDS = 10000
MAX_LEN = 50

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

y = df['sentiment'].map({'negative': 0, 'positive': 1}).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




EMBEDDING_DIM = 100
word_index = tokenizer.word_index
embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))

embedding_index = {}
with open("glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

for word, i in word_index.items():
    if i < MAX_NUM_WORDS:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

results = []
history_dict = {}

def train_evaluate(model_type='rnn', hidden_dim=128):
    print(f"Training {model_type.upper()} with hidden_dim={hidden_dim}")
    model = Sequential()
    model.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False))

    if model_type == 'rnn':
        model.add(SimpleRNN(hidden_dim))
    elif model_type == 'lstm':
        model.add(LSTM(hidden_dim))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=0)
    history_dict[f"{model_type}_{hidden_dim}"] = history

    y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'Model': model_type.upper(),
        'Hidden Dim': hidden_dim,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

for model_type in ['rnn', 'lstm']:
    for hidden_dim in [128, 256, 512]:
        train_evaluate(model_type, hidden_dim)

plt.figure(figsize=(12, 6))
for name, hist in history_dict.items():
    plt.plot(hist.history['val_loss'], label=f"{name} val_loss")
plt.title("Validation Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='Hidden Dim', y='F1-Score', hue='Model')
plt.title("F1-Score Comparison for RNN vs LSTM")
plt.ylim(0.5, 1.0)
plt.grid(True)
plt.show()


