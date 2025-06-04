EyeDiseaseClassifier
A deep learning-based web application for classifying eye diseases (Cataract, Diabetic Retinopathy, Glaucoma, Normal) using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The system integrates a Flask backend for real-time image classification and a user-friendly frontend for seamless interaction. Trained on the Eye Diseases Classification dataset with 4,217 images, the model achieves high accuracy, with results visualized via confusion matrices and classification reports.
Features

Classifies eye diseases into four categories: Cataract, Diabetic Retinopathy, Glaucoma, and Normal.
Uses a CNN model with TensorFlow and Keras, incorporating data augmentation for robust training.
Flask backend serves predictions via a REST API, connected to a frontend for user interaction.
Evaluates model performance with metrics like accuracy, F1 score, and confusion matrix.

Project Structure
EyeDiseaseClassifier/
├── app.py                  # Flask backend for serving the model
├── templates/             # Frontend files (HTML, CSS, JavaScript)
├── scripts/               # Python scripts for model training and evaluation
│   └── train_model.py     # Main script for CNN training
├── model/                 # Directory containing the trained Keras model (best_cnn_model.keras)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

Setup Instructions

Clone the Repository:
git clone https://github.com/AHSAN-MEHMOOD/EyeDiseaseClassifier.git
cd EyeDiseaseClassifier


Set Up Python Environment:

Install Python 3.8+.
Create and activate a virtual environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt




Download Dataset:

Download the Eye Diseases Classification dataset from Kaggle.
Extract it to a dataset/ folder in the project root with the structure: dataset/{cataract,diabetic_retinopathy,glaucoma,normal}/images.


Train the Model:

Run the training script:python scripts/train_model.py


The best model will be saved in model/best_cnn_model.keras.


Run the Flask App:

Start the Flask server:python app.py


Access the web app at http://localhost:5000 (or the specified port).



Dependencies

Python 3.8+
TensorFlow, Keras, NumPy, Matplotlib, Scikit-learn, Flask
See requirements.txt for the full list.

Usage

Upload an eye image via the frontend interface.
The Flask backend processes the image using the trained CNN model and returns the predicted disease category.

Results

Trained on 4,217 images from the Eye Diseases Classification dataset with an 80-20 train-validation split.
Metrics: Validation accuracy, F1 score, and confusion matrix (see train_model.py output).
Visualizations: Confusion matrix plotted using Matplotlib.

Future Improvements

Add support for real-time webcam-based classification.
Enhance frontend with responsive design and additional visualizations.
Optimize model inference for faster predictions.

License
MIT License
