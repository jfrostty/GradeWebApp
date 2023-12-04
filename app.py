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
    
    # Save each feature input as a variable
    guardian_other = float(request.form['guardian_other'])

    famsup_yes = float(request.form['famsup'])
    famsup_no = 1 - famsup_yes  # Complementary variable

    nursery_yes = float(request.form['nursery'])
    nursery_no = 1 - nursery_yes  # Complementary variable

    internet_yes = float(request.form['internet_yes'])
    internet_no = 1 - internet_yes  # Complementary variable

    higher_yes = float(request.form['higher_yes'])
    higher_no = 1 - higher_yes  # Complementary variable

    romantic_yes = float(request.form['romantic_yes'])
    romantic_no = 1 - romantic_yes  # Complementary variable

    activities_yes = float(request.form['activities_yes'])
    activities_no = 1 - activities_yes  # Complementary variable

    paid_yes = float(request.form['paid_yes'])
    paid_no = 1 - paid_yes  # Complementary variable

    schoolsup_yes = float(request.form['schoolsup_yes'])
    schoolsup_no = 1 - schoolsup_yes  # Complementary variable

    freetime = float(request.form['freetime'])
    Walc = float(request.form['Walc'])
    
    # Append variables to a list
    int_features = [
        guardian_other, famsup_no, nursery_yes, internet_yes, higher_yes, higher_no,
        internet_no, romantic_yes, romantic_no, nursery_no, activities_yes, activities_no,
        paid_yes, paid_no, famsup_yes, schoolsup_yes, schoolsup_no, freetime, Walc        
    ]

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

    return render_template('index.html', prediction_text='{}%'.format(grade))

if __name__ == "__main__":
    app.run()
