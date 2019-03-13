import flask
from nltk import download, sent_tokenize, word_tokenize, pos_tag
from nltk.stem.snowball import SnowballStemmer
from WordIndex import WordIndex
from Preprocessor import encodeSentences, encodeStems, encodeChars, pad
from StemCharEmbLstmCrfModel import restore

MAX_LEN = 75
MAX_LEN_CHARS = 15

app = flask.Flask(__name__)
model = None
stemmer = None
wordIndex = None
tagIndex = None
charIndex = None
stemIndex = None

# init function
# load preprocessors and model
def load_model():
    download('punkt')
    download('averaged_perceptron_tagger')
    download('stopwords')

    global stemmer
    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    global wordIndex
    wordIndex = WordIndex("UNK")
    wordIndex.load('models/word_to_index.pickle')

    global tagIndex
    tagIndex = WordIndex("O")
    tagIndex.load('models/tag_to_index.pickle')

    global charIndex
    charIndex = WordIndex("UNK")
    charIndex.load('models/char_to_index.pickle')

    global stemIndex
    stemIndex = WordIndex("UNK")
    stemIndex.load('models/stem_to_index.pickle')

    global model
    model = restore('models/stem_char_lstm_crf.h5')

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    if flask.request.method == "POST":
        request_data = flask.request.get_json()
        text = request_data['text']

        sentences = sent_tokenize(text)

        data["text"] = text
        data["predictions"] = []

        last_idx = 0

        for sentence in sentences:
            words = pos_tag(word_tokenize(sentence))
            encodedInput = encodeSentences([words], wordIndex)
            encodedInput = pad(encodedInput, MAX_LEN, wordIndex.getPadIdx())

            encodedStems = encodeStems([[w[0] for w in words]], stemIndex, stemmer)
            encodedStems = pad(encodedStems, MAX_LEN, stemIndex.getPadIdx())

            encodedChars = encodeChars([words], charIndex, MAX_LEN, MAX_LEN_CHARS)

            prediction = model.predict(encodedInput, encodedStems, encodedChars)
            for w, pred1 in zip(words, prediction[0]):
                tag = tagIndex.getWord(pred1)
                idx = text.find(w[0])
                last_idx = idx + len(w[0])
                data["predictions"].append({
                    "type": tag,
                    "start": idx,
                    "end": last_idx
                })
                

        

        # indicate that the request was a success
        data["success"] = True

	# return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run()