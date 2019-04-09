import flask
from flask_cors import CORS
import logging
import sys, zipfile, os, glob
from threading import Thread

from StreamLogger import StreamToLogger
from Predictor import Predictor
from Linker import Linker

predictor = None
linker = None
model_dir = '/app/models/'

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

def get_model_file(model_dir):
    for file in os.listdir(model_dir):
        if file.endswith(".h5"):
            return file
    return None

def load_model():
    global predictor

    current_model_dir = model_dir + 'current_model/'
    
    try:
        os.stat(current_model_dir)
    except:
        os.mkdir(current_model_dir)

    model_path = get_model_file(current_model_dir)
    if None == model_path:
        zip_ref = zipfile.ZipFile(model_dir+'current_model.zip', 'r')
        zip_ref.extractall(current_model_dir)
        zip_ref.close()

    model_path = get_model_file(current_model_dir)

    predictor = Predictor(current_model_dir + model_path)

def create_app():
    logger.info("* Loading Keras model and Flask starting server..."
    "please wait until server has fully started")

    Thread(target=load_model).start()
    
    global linker
    linker = Linker('bolt://46.101.121.174', 'neo4j', 'get_r1ght')

    application = flask.Flask('esports ner service')
    CORS(application)
    return application

app = create_app()

@app.route('/ready', methods=["GET"])
def ready():
    if predictor != None:
        return flask.Response("", status = 200)
    else:
        return flask.Response("", status = 503)

@app.route("/health", methods=["GET"])
def health():
    return flask.Response("", status=200)

@app.route("/predict", methods=["POST"])
def predict():
    logger.debug('Handling predict request')
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    if flask.request.method == "POST":
        request_data = flask.request.get_json()
        text = request_data['text']

        data["text"] = text
        data["predictions"] = []

        last_idx = 0
        predictions = predictor.predict(text)
        
        for w, tag in predictions:
            if tag != "O":
                idx = text.find(w, last_idx)
                last_idx = idx + len(w)
                if (tag.startswith("B")):
                    data["predictions"].append({
                        "type": tag.split('-')[1],
                        "start": idx,
                        "end": last_idx
                    })
                else :
                    data["predictions"][-1]["end"] = last_idx
                
        for pred in data["predictions"]:
            start_idx = pred["start"]
            end_idx = pred["end"]
            pred["meta"] = linker.getLink(text[start_idx:end_idx] ,pred["type"])

        # indicate that the request was a success
        data["success"] = True

	# return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    
    app.run()