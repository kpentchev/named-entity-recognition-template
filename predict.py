import nltk
import pickle
from Indexer import inverseTagIdx
from Preprocessor import encodeSentences, padSentences
from LstmCrfModel import restore

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


BATCH_SIZE = 32
EPOCHS = 5
MAX_LEN = 75
EMBEDDING = 20

sentence = "Donald Trump announced that he is invading Mexico."

with open('models/word_to_index.pickle', 'rb') as handle:
    word2idx = pickle.load(handle)
 
# Saving Vocab
with open('models/tag_to_index.pickle', 'rb') as handle:
    tag2idx = pickle.load(handle)
    idx2tag = inverseTagIdx(tag2idx)

words = nltk.pos_tag(nltk.word_tokenize(sentence))
encodedInput = encodeSentences([words], word2idx)
encodedInput = padSentences(encodedInput, MAX_LEN, word2idx["PAD"])

model = restore('models/lstm_crf_weights.h5')

prediction = model.predict(encodedInput)

# Visualization asd
print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(words, prediction[0]):
    print("{:15}: {:5}".format(w[0], idx2tag[pred]))