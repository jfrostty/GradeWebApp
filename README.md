# GradeWebApp
A basic web application which utilizes flask to load a Random Forest Machine Learning algorithm to predict a users grade based on some key data points. 

We have 3 subfolders.
- **model**: Model contains our Random Forest NN model which has been saved as a "pickled" file, or a byte stream.
- **templates**: Templates contain our HTML code which is used to display input boxes ,to retrieve user data, and display the models prediction using the POST HTTP method.
- **background**: Background contains the jupyter notebook files used to create our initial models (K-Nearest Neighbors, Neural Network, Decision Tree, Random Forest) post and pre PCA, as well as the datasets (math and Portuguese) post and pre PCA, and the code to compute the principal component analysis. 

**app.py**: Our app.py program creates a flask web application which loads our html and loads our "pickled" random forest ML model. After receiving the user input, our application uses the POST HTTP method to display the output to the user.

**pca_components.py**: Our pca_components.py program is used in tandem with app.py to provide the necessary statistics (means, standard deviations, projection matrix) to perform PCA on the user input data. 

After running app.py, users can visit the web page at http:/127.0.0.1:5000
