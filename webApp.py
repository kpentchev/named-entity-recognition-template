import flask
from flask_cors import CORS
from Predictor import Predictor

app = flask.Flask(__name__)
CORS(app)

predictor = None
    

@app.route("/predict", methods=["POST"])
def predict():
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

        # indicate that the request was a success
        data["success"] = True

	# return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	predictor = Predictor()
	app.run()