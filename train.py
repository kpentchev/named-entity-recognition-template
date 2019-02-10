import tensorflow as tf
import keras as ks 
import numpy as np 
import pandas as pd

from SentenceGetter import SentenceGetter
from WordIndex import WordIndex
from Indexer import buildTagIdx, inverseTagIdx
from Preprocessor import encodeSentences, encodeTags, pad, onehotEncodeTags
from LstmCrfModel import LstmCrfModel

# Hyperparams if GPU is available
MAX_WORDS = 300000
if tf.test.is_gpu_available():
    BATCH_SIZE = 512  # Number of examples used in each iteration
    EPOCHS = 5  # Number of passes through entire dataset
    MAX_LEN = 75  # Max length of review (in words)
    EMBEDDING = 40  # Dimension of word embedding vector

    
# Hyperparams for CPU training
else:
    BATCH_SIZE = 32
    EPOCHS = 5
    MAX_LEN = 75
    EMBEDDING = 20


data = pd.read_csv("/home/kpentchev/data/floyd/ner_dataset.csv", encoding="latin1")
#data = pd.read_csv("/home/kpentchev/data/floyd/teo_tagged_3.csv", encoding="utf-8", delimiter='\t')
data = data.fillna(method="ffill")

print("Number of sentences: ", len(data.groupby(['Sentence #'])))

words = list(set(data["Word"].values))
n_words = len(words)
print("Number of words in the dataset: ", n_words)

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

tagIndex = WordIndex("O")
tagIndex.add(tags)

#print("The word Obama is identified by the index: {}".format(word2idx["Obama"]))
#print("The labels B-geo(which defines Geopraphical Enitities) is identified by the index: {}".format(tag2idx["B-geo"]))

# Get all the sentences
sentences = getter.sentences

encodedSentences = encodeSentences(sentences, wordIndex)
encodedSentences = pad(encodedSentences, MAX_LEN, wordIndex.getPadIdx())

encodedTags = encodeTags(sentences, tagIndex)
encodedTags = pad(encodedTags, MAX_LEN, tagIndex.getPadIdx())
encodedTags = onehotEncodeTags(encodedTags, n_tags)

print('Raw Sample: ', ' '.join([w[0] for w in sentences[0]]))
print('Raw Label: ', ' '.join([w[2] for w in sentences[0]]))
print('After processing, sample:', encodedSentences[0])
print('After processing, labels:', encodedTags[0])


model = LstmCrfModel(MAX_WORDS, n_tags, EMBEDDING, MAX_LEN)

model.train(encodedSentences, encodedTags, BATCH_SIZE, EPOCHS)

model.evaluate(wordIndex, tagIndex)

# Saving Vocab
wordIndex.save('models/word_to_index.pickle')
 
# Saving Vocab
tagIndex.save('models/tag_to_index.pickle')
    
# Saving Model Weight
model.save('models/lstm_crf_weights.h5')
