import tensorflow as tf
import keras as ks 
import numpy as np 
import pandas as pd

import nltk
from nltk.stem.snowball import SnowballStemmer

from SentenceGetter import SentenceGetter
from WordIndex import WordIndex
from Indexer import buildTagIdx, inverseTagIdx
from Preprocessor import encodeSentences, encodeStems, encodeChars, encodeTags, pad, onehotEncodeTags
from LstmCrfModel import LstmCrfModel
from CharEmbLstmCrfModel import CharEmbLstmCrfModel
from StemCharEmbLstmCrfModel import StemCharEmbLstmCrfModel


nltk.download('stopwords')
# Hyperparams if GPU is available
MAX_WORDS = 40000
if tf.test.is_gpu_available():
    BATCH_SIZE = 512  # Number of examples used in each iteration
    EPOCHS = 5  # Number of passes through entire dataset
    MAX_LEN = 75  # Max length of review (in words)
    EMBEDDING = 40  # Dimension of word embedding vector
    EMBEDDING_WORD = 20

    
# Hyperparams for CPU training
else:
    BATCH_SIZE = 32
    EPOCHS = 15
    MAX_LEN = 75
    MAX_LEN_CHARS = 15
    EMBEDDING = 40
    EMBEDDING_WORD = 20


#data = pd.read_csv("/Users/kpentchev/Downloads/ner_dataset.csv", encoding="latin1")
#data = pd.read_csv("/Users/kpentchev/data/ner_2019_03_11_no_med.csv", encoding="utf-8", delimiter='\t')
data = pd.read_csv("/home/kpentchev/data/floyd/ner_2019_02_27_fixed.csv", encoding="utf-8", delimiter='\t')
data = data.fillna(method="ffill")

print("Number of sentences: ", len(data.groupby(['Sentence #'])))

words = list(set(data["Word"].values))
n_words = len(words)
print("Number of words in the dataset: ", n_words)

stemmer = SnowballStemmer("english", ignore_stopwords=True)
stems = list(set(stemmer.stem(word) for word in words))

tags = list(set(data["Tag"].values))
print("Tags:", tags)
n_tags = len(tags)
print("Number of Tags: ", n_tags)

#print("What the dataset looks like:")
# Show the first 10 rows
#print(data.head(n=10))

getter = SentenceGetter(data)
#sent = getter.get_next()
#print('This is what a sentence looks like:')
#print(sent)

wordIndex = WordIndex("UNK")
wordIndex.add(words)

stemIndex = WordIndex("UNK")
stemIndex.add(stems)

tagIndex = WordIndex("O")
tagIndex.add(tags)

chars = set([w_i for w in words for w_i in w])
n_chars = len(chars)
print(chars)

charIndex = WordIndex("UNK")
charIndex.add(chars)

#print("The word Obama is identified by the index: {}".format(word2idx["Obama"]))
#print("The labels B-geo(which defines Geopraphical Enitities) is identified by the index: {}".format(tag2idx["B-geo"]))

# Get all the sentences
sentences = getter.sentences

encodedSentences = encodeSentences(sentences, wordIndex)
encodedSentences = pad(encodedSentences, MAX_LEN, wordIndex.getPadIdx())

encodedStems = encodeStems(sentences, stemIndex, stemmer)
encodedStems = pad(encodedStems, MAX_LEN, stemIndex.getPadIdx())

encodedTags = encodeTags(sentences, tagIndex)
encodedTags = pad(encodedTags, MAX_LEN, tagIndex.getPadIdx())
encodedTags = onehotEncodeTags(encodedTags, n_tags)

encodedChars = encodeChars(sentences, charIndex, MAX_LEN, MAX_LEN_CHARS)

print('n encoded sentences: {}'.format(len(encodedSentences)))
print('n encoded chars: {}'.format(len(encodedChars)))

#model = CharEmbLstmCrfModel(MAX_WORDS, n_chars, n_tags, EMBEDDING, EMBEDDING_WORD, MAX_LEN, MAX_LEN_CHARS)
model = StemCharEmbLstmCrfModel(MAX_WORDS, n_chars, n_tags, EMBEDDING, EMBEDDING_WORD, MAX_LEN, MAX_LEN_CHARS)

model.train(encodedSentences, encodedStems, encodedChars, encodedTags, BATCH_SIZE, EPOCHS, 0.1)

model.evaluate(wordIndex, tagIndex)

#model = LstmCrfModel(MAX_WORDS, n_tags, EMBEDDING, MAX_LEN)

#model.train(encodedSentences, encodedTags, BATCH_SIZE, EPOCHS, 0.1)

#model.evaluate(wordIndex, tagIndex)

# Saving word index
wordIndex.save('models/word_to_index.pickle')

# Saving tag index
stemIndex.save('models/stem_to_index.pickle')

# Saving tag index
tagIndex.save('models/tag_to_index.pickle')

# Saving character index
charIndex.save('models/char_to_index.pickle')
    
# Saving Model
model.save('models/stem_char_lstm_crf.h5')
