from flask import Flask, request, jsonify
import pickle


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

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error='The requested URL was not found on the server'), 404

if __name__ == "__main__":
	app.run(host ='0.0.0.0', port = 5000, debug = True)
