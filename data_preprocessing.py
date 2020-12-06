import numpy as np
import xml.etree.ElementTree as ET

def load_embeddings():
    f = open('embeddings/glove.6B.100d.txt', 'r', encoding='utf8')
    lines = f.readlines()
    word2index = {}
    embeddings = np.zeros(shape=(len(lines) + 1, len(lines[0].split()[1:])), dtype=np.float32)
    for idx, line in enumerate(lines):
        line = line.split()
        word2index[line[0]] = len(word2index) + 1
        embeddings[idx + 1] = np.array(line[1:], dtype=np.float32)
    return embeddings, word2index


def load_dataset(word2index):
    root = ET.parse('dataset/semcor.data.xml').getroot()
    pos2index = {}
    lemma2index = {'unk': 1}
    ambiguous_words = set()
    synset2index = {'_': 1}
    f = open('dataset/semcor.gold.key.bnids.txt', 'r', encoding='utf8')
    for sentence in root.findall('text/sentence'):
        for word in sentence:
            lemma = word.attrib['lemma'].lower()
            tag = word.attrib['pos'].lower()
            if lemma not in lemma2index:
                lemma2index[lemma] = len(lemma2index) + 1
            if tag not in pos2index:
                pos2index[tag] = len(pos2index) + 1
            if 'id' in word.attrib:
                synset = f.readline().split()[1]
                if lemma in word2index:
                    ambiguous_words.add(word2index[lemma])
                if synset not in synset2index:
                    synset2index[synset] = len(synset2index) + 1
    return lemma2index, pos2index, synset2index, ambiguous_words


def load_sentibabelnet():
    f = open('dataset/SentiBabelNet-EN.dict', 'r', encoding='utf8')
    synset2score = {}
    for line in f.readlines():
        if line.startswith('bn'):
            line = line.split('\t')
            syn = line[0]
            pos_score = float(line[2][2:])
            neg_score = float(line[3][1:])
            synset2score[syn] = np.array([pos_score, neg_score])
    return synset2score
