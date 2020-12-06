import numpy as np
import re
import tensorflow as tf

class SentimentClassifier():

    def __init__(self):
        #load Glove embeddings
        f = open('embeddings/glove.6B.100d.txt', 'r', encoding='utf8')
        lines = f.readlines()
        self.word2index = {}
        embeddings = np.zeros(shape=(len(lines) + 1, len(lines[0].split()[1:])), dtype=np.float32)
        for idx, line in enumerate(lines):
            line = line.split()
            self.word2index[line[0]] = len(self.word2index) + 1
            embeddings[idx + 1] = np.array(line[1:], dtype=np.float32)

        #load the model backup
        self.model = self.__load_model(embeddings)

        self.label = ['positive', 'negative']


    def __load_model(self, embeddings):
        word_ids = tf.keras.Input([None], dtype=tf.int32)
        emb = tf.keras.layers.Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], mask_zero=True, trainable=False)(word_ids)
        bid = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False))(emb)
        dropout = tf.keras.layers.Dropout(0.5)(bid)
        bn = tf.keras.layers.BatchNormalization()(dropout)
        dense = tf.keras.layers.Dense(1, activation='sigmoid')(bn)

        model = tf.keras.Model(inputs=word_ids, outputs=dense)
        model.load_weights('models/sentiment_classifier_sigmoid.h5')
        return model

    def tokenize(self, sentence):
        #remove HTML tags
        s = re.sub('<[^>]*>', ' ', sentence.lower())
        #remove symbols
        s = re.sub('[^a-z0-9\']', ' ', s)
        s = re.sub('([\'])([ ]*)', r' \1', s)
        return s.strip().split()

    def predict_proba(self, sentence):
        sentence = [self.word2index[word] for word in self.tokenize(sentence)]
        return self.model.predict([[sentence]]).reshape(-1)[0]

    def predict(self, sentence):
        prediction = int(round(self.predict_proba(sentence)))
        return 'positive' if prediction == 0 else 'negative'
