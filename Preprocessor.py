import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk.corpus import wordnet
from tqdm import tqdm

whitespaceMatch = re.compile(r'\s+')
numberStem = '###NUM###'

def isNumber(text):
    cleanText = text.replace('.','',1).replace('"', '')
    if cleanText.startswith('-'):
        cleanText = cleanText[1:]
    cleanText = whitespaceMatch.sub('', cleanText)
    return cleanText.isdigit()

def encodeSentences(sentences, word2idx):
    # Convert each sentence from list of Token to list of word_index
    encoded = [[word2idx.getIdx(w[0]) for w in s] for s in sentences]
    return encoded

def encodeStems(sentences, stem2idx, stemmer):
    # Convert each sentence from list of Token to list of word_index
    encoded = [[stem2idx.getIdx(numberStem if isNumber(w[0]) else stemmer.stem(w[0])) for w in s] for s in tqdm(sentences)]
    return encoded

def encodeLemmas(sentences, lemma2idx, lemmatizer):
    # Convert each sentence from list of Token to list of word_index
    encoded = [[lemma2idx.getIdx(numberStem if isNumber(w[0]) else lemmatizer.lemmatize(w[0], __getWordnetPos(w[1]))) for w in s] for s in tqdm(sentences)]
    return encoded

def encodeChars(sentences, charIndex, maxLengthSentence, maxLengthWord):
    encodedChars = []
    for sentence in sentences:
        sent_seq = []
        for i in range(maxLengthSentence):
            word_seq = []
            for j in range(maxLengthWord):
                try:
                    word_seq.append(charIndex.getIdx(sentence[i][0][j]))
                except:
                    word_seq.append(charIndex.getPadIdx())
            sent_seq.append(word_seq)
        encodedChars.append(np.array(sent_seq))
    return encodedChars

def pad(sentences, maxLength, pad):
    # Padding each sentence to have the same lenght
    padded = pad_sequences(maxlen=maxLength, sequences=sentences, padding="post", value=pad)
    return padded

def encodeTags(tags, tag2idx):
    # Convert Tag/Label to tag_index
    encoded = [[tag2idx.getIdx(w[2]) for w in s] for s in tags]
    return encoded

def onehotEncodeTags(tags, nTags):
    # One-Hot encode
    hotencoded = [to_categorical(i, num_classes=nTags+1) for i in tags]  # n_tags+1(PAD)
    return hotencoded

def __getWordnetPos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN