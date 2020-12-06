import tensorflow as tf
import data_preprocessing as dp
import re
import numpy as np

class SentimentAnalyzer():

    def __init__(self):
        self.__embeddings, self.word2index = dp.load_embeddings()
        self.lemma2index, self.pos2index, self.synset2index, self.ambiguous_words = dp.load_dataset(self.word2index)
        self.multitask_model = self.__load_multitask_model()
        self.disambiguation_model = self.__load_disambiguation_model()
        self.index2lemma = {v: k for k, v in self.lemma2index.items()}
        self.index2synset = {v: k for k, v in self.synset2index.items()}
        self.synset2score = dp.load_sentibabelnet()

    def __load_multitask_model(self):
        hidden_size = 100

        word_ids = tf.keras.Input([None], dtype=tf.int32)

        pretrained_emb = tf.keras.layers.Embedding(self.__embeddings.shape[0], self.__embeddings.shape[1], weights=[self.__embeddings], mask_zero=True, trainable=False)(word_ids)

        bid = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, return_sequences=True))(pretrained_emb)

        lemmas_scores = tf.keras.layers.Dense(len(self.lemma2index) + 1)(bid)
        pos_scores = tf.keras.layers.Dense(len(self.pos2index) + 1)(bid)

        model = tf.keras.Model(inputs=word_ids, outputs=[lemmas_scores, pos_scores])
        model.load_weights('models/multitask.h5')

        return model

    def __load_disambiguation_model(self):
        hidden_size = 100

        word_ids = tf.keras.Input([None], dtype=tf.int32)
        pos_ids = tf.keras.Input([None], dtype=tf.int32)
        flags = tf.keras.Input([None], dtype=tf.int32)

        pretrained_emb = tf.keras.layers.Embedding(self.__embeddings.shape[0], self.__embeddings.shape[1], weights=[self.__embeddings], mask_zero=True, trainable=False)(word_ids)
        pos_emb = tf.keras.layers.Embedding(len(self.pos2index) + 1, 50, mask_zero=True)(pos_ids)
        final_emb = tf.keras.layers.Concatenate(axis=-1)([pretrained_emb, pos_emb, tf.cast(tf.expand_dims(flags, axis=-1), tf.float32)])

        bid = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, return_sequences=True))(final_emb)
        scores = tf.keras.layers.Dense(len(self.synset2index) + 1)(bid)

        model = tf.keras.Model(inputs=[word_ids, pos_ids, flags], outputs=scores)
        model.load_weights('models/wsd.h5')

        return model

    def tokenize(self, sentence):
        s = re.sub('([^a-z0-9])', r' \1 ', sentence.lower())
        s = re.sub('["]', r' ', s)
        return re.sub('([\'])([ ]*)', r' \1', s).strip().split()

    def predict_lemmas_and_pos_tags(self, sentence):
        sent = [self.word2index[w] if w in self.word2index else self.word2index['unk'] for w in self.tokenize(sentence)]
        lem, ps = self.multitask_model.predict(sent)
        lem = lem.argmax(axis=-1).reshape(-1)
        ps = ps.argmax(axis=-1).reshape(-1)
        return lem, ps

    def predict_synsets(self, sentence=None, lemmas=None, pos_tags=None):
        if sentence:
            lemmas, pos_tags = self.predict_lemmas_and_pos_tags(sentence)
        words = [self.word2index[self.index2lemma[w]] if self.index2lemma[w] in self.word2index else self.word2index['unk'] for w in lemmas]
        synsets = self.disambiguation_model.predict([words, pos_tags, [1 if w in self.ambiguous_words else 0 for w in words]])
        return [self.index2synset[w] for w in synsets.argmax(axis=-1).reshape(-1)]

    def predict_proba(self, sentence, from_synsets=False):
        if from_synsets:
            synsets = sentence
        else:
            lemmas, pos_tags = self.predict_lemmas_and_pos_tags(sentence)
            synsets = self.predict_synsets(lemmas=lemmas, pos_tags=pos_tags)
        try:
            stack = np.vstack([self.synset2score[synset] for synset in synsets if synset in self.synset2score])
            tot = np.sum(stack)
            return np.sum(stack, axis=0) / tot if tot > 0 else np.array([0., 0.])
        except:
            return np.array([0., 0.])

    def predict(self, sentence):
        prediction = self.predict_proba(sentence)
        if prediction[0] == prediction[1]:
            return 'Neutral'
        return 'Positive' if prediction.argmax(axis=-1) == 0 else 'Negative'



