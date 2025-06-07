# Fire Classification using VGG16 Transfer Learning

A deep learning project that classifies images as containing fire or not using transfer learning with VGG16 pre-trained model. This project achieves high accuracy in fire detection through convolutional neural networks.

## 🔥 Project Overview

This project implements a binary image classifier to detect fire in images using:
- **Transfer Learning** with VGG16 pre-trained on ImageNet
- **Data Augmentation** for improved model generalization
- **Custom fully connected layers** for fire-specific feature learning
- **Advanced callbacks** for optimal training

## 📋 Features

- Binary classification (Fire vs Non-Fire)
- Data augmentation for robust training
- Model evaluation with detailed metrics
- Confusion matrix visualization
- Training history plots
- Model saving and loading capabilities
- Single image prediction function

## 🛠️ Technology Stack

- **Python 3.7+**
- **TensorFlow/Keras** - Deep learning framework
- **VGG16** - Pre-trained convolutional neural network
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - Model evaluation metrics
- **NumPy** - Numerical computations

## 📊 Dataset

The project uses the Fire Dataset from Kaggle:
- **Source**: [Fire Dataset by phylake1337](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
- **Structure**: 
  - `fire_images/` - Images containing fire
  - `non_fire_images/` - Images without fire
- **Format**: PNG, JPG, JPEG images
- **Split**: 80% training, 20% validation

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mahdiahmadii/MyUniAIProjects.git
cd MyUniAIProjects
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook or Python script:
```bash
jupyter notebook fire_classification.ipynb
```

## 📁 Project Structure

```
fire-classification/
│
├── fire_classification.ipynb    # Main notebook with complete code
├── requirements.txt            # Project dependencies
├── README.md                  # Project documentation
├── models/                    # Saved models directory
│   └── fire_classification_vgg16_model.h5
├── results/                   # Training results and plots
│   ├── training_history.pkl
│   ├── confusion_matrix.png
│   └── training_plots.png
└── utils/                     # Utility functions
    └── prediction_utils.py
```

## 🔧 Model Architecture

The model uses **VGG16** as the base with custom top layers:

1. **Base Model**: VGG16 (pre-trained on ImageNet, frozen layers)
2. **Global Average Pooling**: Reduces spatial dimensions
3. **Dense Layer 1**: 512 units with ReLU activation
4. **Dropout**: 0.5 rate for regularization
5. **Dense Layer 2**: 256 units with ReLU activation
6. **Dropout**: 0.3 rate for regularization
7. **Output Layer**: 1 unit with sigmoid activation (binary classification)

### Key Parameters
- **Image Size**: 224x224x3
- **Batch Size**: 32
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Binary Crossentropy
- **Epochs**: 50 (with early stopping)

## 📈 Data Augmentation

The training data is augmented with:
- Rotation (±20 degrees)
- Width/Height shifts (±20%)
- Horizontal flipping
- Zoom (±20%)
- Shear transformation (±20%)
- Brightness adjustment (0.8-1.2x)

## 🎯 Model Performance

The model achieves excellent performance on fire detection:

### Training Results
- **Validation Accuracy**: ~95%+
- **Training Accuracy**: ~98%+
- **Precision**: High precision for fire detection
- **Recall**: High recall for fire detection
- **F1-Score**: Balanced performance metric

### Evaluation Metrics
- Confusion Matrix
- Classification Report
- Precision, Recall, F1-Score
- Specificity and Sensitivity

## 📊 Visualizations

The project includes several visualizations:
1. **Training History**: Accuracy and loss curves
2. **Confusion Matrix**: Model performance breakdown
3. **Class Distribution**: Dataset balance analysis

## 🔮 Usage Examples

### Training the Model
```python
# The main notebook contains all training code
# Simply run all cells to train the model
```

### Making Predictions
```python
# Load the trained model
model = tf.keras.models.load_model('fire_classification_vgg16_model.h5')

# Predict single image
predicted_class, confidence = predict_single_image(
    model, 
    'path_to_image.jpg', 
    ['non_fire_images', 'fire_images']
)
print(f"Prediction: {predicted_class} (Confidence: {confidence:.4f})")
```


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle** for providing the fire dataset
- **TensorFlow team** for the excellent deep learning framework
- **VGG authors** for the pre-trained model architecture
- **Open source community** for various libraries used

## 📞 Contact

**Mahdi Ahmadi** - [GitHub Profile](https://github.com/mahdiahmadii)

Project Link: [https://github.com/mahdiahmadii/MyUniAIProjects](https://github.com/mahdiahmadii/MyUniAIProjects)

---

⭐ **Star this repository if you found it helpful!**