// Handwritten Digit Recognition

A machine learning mini-project that recognizes handwritten digits from images using a Convolutional Neural Network (CNN), with a Streamlit web interface.

// Project Overview

The application supports:

- Single handwritten digit recognition
- Multiple handwritten digit recognition from one image
- Per-digit confidence display
- Image preprocessing and segmentation using OpenCV
- Typed arithmetic expression solving as a separate feature

 / Important: The CNN is trained for digits '0–9'. The current equation solver accepts typed expressions; it is not yet a full handwritten equation recognizer.

// Model

The final model is a CNN trained on the MNIST handwritten digit dataset.

/// Input

- Grayscale image
- Resized to '64 × 64'
- Pixel values normalized to '0–1'

/// CNN

- Data augmentation
- Conv2D: 32 filters
- MaxPooling
- Conv2D: 64 filters
- MaxPooling
- Conv2D: 128 filters
- MaxPooling
- Flatten
- Dense: 128
- Dropout: 0.30
- Softmax output: 10 classes

/// Training

- Optimizer: Adam
- Learning rate: '0.001'
- Loss: Sparse Categorical Crossentropy
- Epochs: '10'
- Batch size: '128'
- Validation split: '10%'

/// Evaluation

The final CNN achieved:

99.33% test accuracy on MNIST

This benchmark result does not guarantee the same accuracy on arbitrary real-world photographs.

// System Pipeline

'''text
Uploaded Image
      ↓
Image Preprocessing
      ↓
Character Segmentation
      ↓
64×64 Digit Images
      ↓
CNN Prediction
      ↓
Confidence
      ↓
Recognized Number
'''

For example:

'''text
Image: 97521
       ↓
[9] [7] [5] [2] [1]
       ↓
CNN
       ↓
97521
'''

// Project Structure

'''text
DigitRecognition/
│
├── app.py
├── predict.py
├── equation.py
├── train_model.py
├── evaluation.py
├── requirements.txt
│
└── models/
    ├── digit_model.pkl
    └── handwritten_digit_cnn.keras
'''

// Installation

Create and activate a virtual environment:

'''bash
python -m venv venv
'''

Windows:

'''bash
venv\Scripts\activate
'''

Install dependencies:

'''bash
python -m pip install -r requirements.txt
'''

If OpenCV or TensorFlow is missing:

'''bash
python -m pip install opencv-python tensorflow
'''

// Run the Application

From the project root:

'''bash
streamlit run app.py
'''

Then open the local Streamlit address displayed in the terminal.

// How to Use

/// Single Digit

1. Select Single Digit.
2. Upload an image containing one handwritten digit.
3. Click the prediction button.
4. View the predicted digit and confidence.

/// Multiple Digits

1. Select Multiple Digits.
2. Upload one image containing several handwritten digits.
3. For best results, use plain white paper.
4. Keep digits reasonably separated.
5. The system segments the characters from left to right.
6. The CNN predicts each character and reconstructs the number.

Example:

'''text
5   2   7
'''

Expected output:

'''text
527
'''

// Practical Testing

Clean handwritten digits on plain white paper worked well during testing.

Ruled notebook paper and faint background marks can cause segmentation errors because the image-processing stage may detect background components as characters. The segmentation pipeline was improved to focus on the main writing line and filter small components, but real-world OCR remains an area for improvement.

// Equation Solver

The project also contains an arithmetic expression module.

Examples:

'''text
25 + 13
'''

'''text
12 * 4
'''

'''text
100 / 5
'''

The current equation feature uses typed input. Handwritten operators such as '+', '-', '×', and '÷' are not recognized by the digit CNN.

// Baseline Model

An initial Random Forest classifier was also developed as a baseline. It achieved approximately 97.22% test accuracy in the initial experiment.

The CNN was selected for the final image-recognition pipeline because convolutional layers are designed to learn spatial features from images.

// Limitations

- The model recognizes digits '0–9', not arbitrary mathematical symbols.
- MNIST-style images are cleaner than many real photographs.
- Lighting, shadows, paper texture and notebook lines can affect recognition.
- Multi-digit recognition depends on successful character segmentation.
- The current equation solver is typed rather than handwritten.

// Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pillow
- Streamlit
- scikit-learn
- Matplotlib

// Conclusion

Project: Handwritten Digit Recognition  
Category: Machine Learning / Computer Vision  
Interface: Streamlit  
Primary Model: Convolutional Neural Network  
Dataset: MNIST  
Final MNIST Test Accuracy: 99.33%

