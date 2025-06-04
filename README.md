# EyeDiseaseClassifier

A deep learning web app for classifying eye diseases (Cataract, Diabetic Retinopathy, Glaucoma, Normal) using a CNN built with TensorFlow and Keras. Features a Flask backend for real-time image classification and a user-friendly frontend. Trained on the Eye Diseases Classification dataset with 4,217 images, achieving high accuracy with visualizations.

## Features

- Classifies four eye diseases: Cataract, Diabetic Retinopathy, Glaucoma, Normal
- CNN model with TensorFlow/Keras and data augmentation
- Flask backend with REST API and frontend for user interaction
- Performance metrics: accuracy, F1 score, confusion matrix

## Project Structure

```
EyeDiseaseClassifier/
├── app.py                  # Flask backend for serving the model
├── templates/              # Frontend files (HTML, CSS, JavaScript)
├── scripts/                # Python scripts for model training
│   └── train_model.py      # Main script for CNN training
├── model/                  # Trained Keras model (best_cnn_model.keras)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/AHSAN-MEHMOOD/EyeDiseaseDetection.git
   cd EyeDiseaseClassifier
   ```

2. **Set Up Python Environment**:

   - Install Python 3.8+.

   - Create and activate a virtual environment:

     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```

   - Install dependencies:

     ```bash
     pip install -r requirements.txt
     ```

3. **Download Dataset**:

   - Download the Eye Diseases Classification dataset from Kaggle.
   https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification
   - Extract to `dataset/` with structure: `dataset/{cataract,diabetic_retinopathy,glaucoma,normal}/images`.

4. **Train the Model**:

   - Run the training script:

     ```bash
     python scripts/train_model.py
     ```

   - Best model saved in `model/best_cnn_model.keras`.

5. **Run the Flask App**:

   - Start the Flask server:

     ```bash
     python app.py
     ```

   - Access the web app at `http://localhost:5000`.

## Dependencies

- Python 3.8+
- TensorFlow, Keras, NumPy, Matplotlib, Scikit-learn, Flask
- Full list in `requirements.txt`

## Usage

- Upload an eye image via the frontend.
- Flask backend processes the image and returns the predicted disease.

## Results

- Trained on 4,217 images with 80-20 train-validation split
- Metrics: Validation accuracy, F1 score, confusion matrix
- Visualizations: Confusion matrix via Matplotlib

## Future Improvements

- Real-time webcam-based classification
- Responsive frontend design
- Faster model inference

## License

MIT License