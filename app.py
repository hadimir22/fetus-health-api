from flask import Flask, request
import joblib
import pickle


app = Flask(__name__)

# model = joblib.load('./classifier_model.joblib')
# print(model)
#
# @app.route('/predict', methods = ['POST'])

model = pickle.load(open('model.pkl','rb'))
print('model iss', model)
@app.route('/')
def hello():
	try:
		# data = request.form['data']
		data = [[1,2,3,0,0,0,7,8,9,10,11]]
		prediction = model.predict_proba(data)
		print(prediction)
		return 'prediction'
	except:
		return 'something went wrong'


if __name__ == "__main__":
	app.run(host ='0.0.0.0', port = 5000, debug = True)

