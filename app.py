import numpy as np
from flask import Flask, request, render_template
import pickle

# PCA values in pca_components.py
from pca_components import pca_mean, pca_stdev, projection_matrix
from pca_components import pca_grade_mean, pca_grade_stdev

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model/finalized_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]

    # calculate user input values after PCA
    features = np.array(int_features)
    features_standardized = (features - pca_mean) / pca_stdev
    features_pca = np.dot(features_standardized, projection_matrix.T[:,:19])

    prediction = model.predict(features_pca.reshape(1, -1))
    output = prediction[0]

    # revert grade back from standardized and calculate % grade
    grade = (output * pca_grade_stdev) + pca_grade_mean
    grade = grade / 20 * 100
    grade = round(grade, 2)

    return render_template('index.html', prediction_text='G3: {}%'.format(grade))

if __name__ == "__main__":
    app.run()
