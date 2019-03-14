import pickle
from flask import request
from flask import Flask
app = Flask(__name__)

# Trained iris classifier using svm
filename = 'model.pkl' 

@app.route("/")
def main():
	return "<h3>1, 2, 1 , 2 ... is this thing on?</h3>"
 
@app.route("/predict/", methods=['GET'])
def predict():
	# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
	# Example:  http://0.0.0.0:5000/predict/?sepal_length=6.9&&sepal_width=3.1&&petal_length=4.9&&petal_width=1.5
	sepal_length = request.args.get('sepal_length')
	sepal_width = request.args.get('sepal_width')
	petal_length = request.args.get('petal_length')
	petal_width = request.args.get('petal_width')
	feature_set = [sepal_length, sepal_width, petal_length, petal_width]
	target_names = ['setosa', 'versicolor', 'virginica']
	# load the model from disk
	model = pickle.load(open(filename, 'rb'))
	result = model.predict([feature_set])
	
	return target_names[int(result)]

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000, debug=True)