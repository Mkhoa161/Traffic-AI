# Traffic Sign Classification Using Convolutional Neural Networks

This project implements a **traffic sign classification** system using a Convolutional Neural Network (CNN) built with **TensorFlow**. The model is trained to recognize traffic signs from a dataset containing images of various road signs. The project follows a typical machine learning pipeline including data loading, model training, evaluation, and saving the trained model.

The goal of this project is to create a model capable of accurately predicting traffic sign categories from input images, which can be extended for use in autonomous driving or other applications requiring traffic sign recognition.

## Key Features
- **Traffic Sign Classification**: Uses a CNN to classify images of traffic signs into predefined categories.
- **Data Preprocessing**: The dataset is processed into images of fixed size (30x30), and labels are one-hot encoded for classification.
- **Model Architecture**: Utilizes a Convolutional Neural Network (CNN) architecture with convolutional, pooling, and dense layers.
- **Training and Evaluation**: The model is trained using the Adam optimizer and categorical crossentropy loss. Performance is evaluated on a test set.
- **Model Saving**: The trained model can be saved to a file for later use.

## Requirements

To run this project, you need to have the following libraries installed:

- `tensorflow`
- `numpy`
- `opencv-python`
- `scikit-learn`

You can install them using pip:

```bash
pip install tensorflow numpy opencv-python scikit-learn
