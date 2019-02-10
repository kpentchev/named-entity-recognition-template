import nltk
import pickle
from WordIndex import WordIndex
from Indexer import inverseTagIdx
from Preprocessor import encodeSentences, pad
from LstmCrfModel import restore

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


BATCH_SIZE = 32
EPOCHS = 5
MAX_LEN = 75
EMBEDDING = 20

sentence = "Riot Games announced a partnership deal with Toyota."

wordIndex = WordIndex("UNK")
wordIndex.load('models/word_to_index.pickle')

tagIndex = WordIndex("O")
tagIndex.load('models/tag_to_index.pickle')


words = nltk.pos_tag(nltk.word_tokenize(sentence))
encodedInput = encodeSentences([words], wordIndex)
encodedInput = pad(encodedInput, MAX_LEN, wordIndex.getPadIdx())

model = restore('models/lstm_crf_weights.h5')

prediction = model.predict(encodedInput)

# Visualization asd
print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(words, prediction[0]):
    print("{:15}: {:5}".format(w[0], tagIndex.getWord(pred)))

wordIndex.save('models/word_to_index.pickle')