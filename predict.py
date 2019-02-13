import nltk
import pickle
from WordIndex import WordIndex
from Indexer import inverseTagIdx
from Preprocessor import encodeSentences, encodeChars, pad
#from LstmCrfModel import restore
from CharEmbLstmCrfModel import restore

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


BATCH_SIZE = 32
EPOCHS = 5
MAX_LEN = 75
MAX_LEN_CHARS = 10
EMBEDDING = 20

#sentence = "Valve announced a partnership deal with BMW for their Dota 2 The International event."



wordIndex = WordIndex("UNK")
wordIndex.load('models/word_to_index.pickle')

tagIndex = WordIndex("O")
tagIndex.load('models/tag_to_index.pickle')

charIndex = WordIndex("UNK")
charIndex.load('models/char_to_index.pickle')

model = restore('models/lstm_crf_weights.h5')

text = 'Dota 2 developer Psyonix and Turner Sports’ ELEAGUE esports brand have announced a multi-faceted business, event, and broadcasting deal around the Dota 2 Championship Series (RLCS) and collegiate esports. ELEAGUE will produce a multi-part feature series around the upcoming RLCS Season 7, which begins its regular season in April, as well as the future Season 8 to follow. The series will cover big moments and behind-the-scenes stories from each season and will debut on the TBS cable network “later in 2019,” according to a release. Additionally, Turner Sports’ ad sales team will oversee all advertising and sponsorships for those two RLCS seasons. Furthermore, ELEAGUE and Psyonix will host a Collegiate Dota 2 (CRL) Spring Invitational event at the NCAA Final Four Fan Fest presented by Capital One, which will be held in Minneapolis from April 5-8. The top four teams from the CRL Spring Season will participate in the exhibition tournament.'

sentences = nltk.sent_tokenize(text)

print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
    
for sentence in sentences:
    words = nltk.pos_tag(nltk.word_tokenize(sentence))
    encodedInput = encodeSentences([words], wordIndex)
    encodedInput = pad(encodedInput, MAX_LEN, wordIndex.getPadIdx())

    encodedChars = encodeChars([words], charIndex, MAX_LEN, MAX_LEN_CHARS)

    prediction = model.predict(encodedInput, encodedChars)

    # Visualization asd
    for w, pred in zip(words, prediction[0]):
        print("{:15}: {:5}".format(w[0], tagIndex.getWord(pred)))

wordIndex.save('models/word_to_index.pickle')