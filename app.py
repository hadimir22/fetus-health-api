from flask import Flask, request, jsonify
import pickle
import numpy as np


app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))


@app.route('/predict', methods = ['POST'])
def predict():
	try:
		data = request.get_json()
		prediction = model.predict_proba([data['features']])
		print(prediction)
		output = {'predictions': prediction.tolist()[0]}
		return jsonify( output	)
	except NameError:
		print(NameError)
		return 'something went wrong'


if __name__ == "__main__":
	app.run(host ='0.0.0.0', port = 5000, debug = True)

