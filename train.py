import tensorflow as tf
import keras as ks 
import numpy as np 
import pandas as pd

from SentenceGetter import SentenceGetter
from WordIndex import WordIndex
from Indexer import buildTagIdx, inverseTagIdx
from Preprocessor import encodeSentences, encodeTags, pad, onehotEncodeTags
from LstmCrfModel import LstmCrfModel
from CharEmbLstmCrfModel import CharEmbLstmCrfModel

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
    EPOCHS = 10
    MAX_LEN = 75
    MAX_LEN_CHARS = 10
    EMBEDDING = 20
    EMBEDDING_WORD = 10


data = pd.read_csv("/Users/kpentchev/Downloads/ner_dataset.csv", encoding="latin1")
#data = pd.read_csv("/Users/kpentchev/git/tf_ner/teo_tagged_6.csv", encoding="utf-8", delimiter='\t')
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

chars = set([w_i for w in words for w_i in w])
n_chars = len(chars)
print(chars)

#print("The word Obama is identified by the index: {}".format(word2idx["Obama"]))
#print("The labels B-geo(which defines Geopraphical Enitities) is identified by the index: {}".format(tag2idx["B-geo"]))

# Get all the sentences
sentences = getter.sentences

encodedSentences = encodeSentences(sentences, wordIndex)
encodedSentences = pad(encodedSentences, MAX_LEN, wordIndex.getPadIdx())

encodedTags = encodeTags(sentences, tagIndex)
encodedTags = pad(encodedTags, MAX_LEN, tagIndex.getPadIdx())
encodedTags = onehotEncodeTags(encodedTags, n_tags)

char2idx = {c: i + 2 for i, c in enumerate(chars)}
char2idx["UNK"] = 1
char2idx["PAD"] = 0

encodedChars = []
for sentence in sentences:
    sent_seq = []
    for i in range(MAX_LEN):
        word_seq = []
        for j in range(MAX_LEN_CHARS):
            try:
                word_seq.append(char2idx.get(sentence[i][0][j]))
            except:
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    encodedChars.append(np.array(sent_seq))

print('n encoded sentences: {}'.format(len(encodedSentences)))
print('n encoded chars: {}'.format(len(encodedChars)))

model = CharEmbLstmCrfModel(MAX_WORDS, n_chars, n_tags, EMBEDDING, EMBEDDING_WORD, MAX_LEN, MAX_LEN_CHARS)

model.train(encodedSentences, encodedChars, encodedTags, BATCH_SIZE, EPOCHS, 0.1)

#model = LstmCrfModel(MAX_WORDS, n_tags, EMBEDDING, MAX_LEN)

#model.train(encodedSentences, encodedTags, BATCH_SIZE, EPOCHS, 0.1)

#model.evaluate(wordIndex, tagIndex)

# Saving Vocab
wordIndex.save('models/word_to_index.pickle')
 
# Saving Vocab
tagIndex.save('models/tag_to_index.pickle')
    
# Saving Model Weight
model.save('models/lstm_crf_weights.h5')
