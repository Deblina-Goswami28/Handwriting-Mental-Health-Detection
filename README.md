# Handwriting-Mental-Health-Detection
Machine learning project to detect mental health conditions (depression, anxiety, normal) from handwriting samples using SVM.  A handwriting-based classification system that extracts features from handwriting images and predicts conditions using Support Vector Machine (SVM).Technologies used (Python, SVM, scikit-learn, etc.)


##Features
- Extracts features from handwriting images
- Classifies handwriting into mental health conditions
- Uses Python, scikit-learn, and SVM

#Installation
```bash
git clone https://github.com/Deblina-Goswami28/Handwriting-Mental-Health-Detection.git
cd Handwriting-Mental-Health-Detection
pip install numpy pandas opencv-python scikit-learn matplotlib joblib


#Run
python train_svm_model.py   # Train the model
python predict_handwriting.py --image sample.jpg  # Predict from handwriting image

#output
depression or normal or anxiety
