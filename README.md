# Emotion Detection Model

A deep learning model for detecting emotions in text using DistilBERT and fine-tuning with emotion classification. The model can detect emotions and transform text to convey different emotional tones.

## Project Overview
This project implements a text emotion detection system with the following capabilities:
- Emotion classification in text using fine-tuned DistilBERT
- Text tone transformation using LangChain and GPT-4
- Support for multiple languages
- Handling class imbalance through data augmentation

## Framework
![Framework](framework.png)

The system consists of two main components:
1. **Emotion Detection Model**: Fine-tuned DistilBERT for classifying text into 6 emotion categories
2. **Text Transformation Pipeline**: Using LangChain and GPT-4 to modify text emotional tone

## Dataset
The project uses the [Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion) containing:
- 16,000 training examples
- 6 emotion categories
- English language text samples
- Labels: sadness, joy, love, anger, fear, surprise

## Results
The model achieves:
- Training loss: 0.384
- Training samples per second: 122.536
- Training steps per second: 15.321
- Total epochs: 2.0

## Prerequisites
- Python 3.10+
- pandas==1.5.3
- numpy==1.23.5
- datasets==2.10.1
- transformers==4.27.1
- torch==1.13.1
- matplotlib==3.6.3
- scikit-learn==1.2.2
- nltk==3.8.1
- nlpaug==1.1.11
- langchain==0.0.101
- python-dotenv==1.0.0
- typing-extensions==4.5.0

## Install dependencies
pip install -r requirements.txt

## Train the model
jupyter notebook Emotion_detection_model_training.ipynb

## Test the model
Use the test.ipynb file to test custom input texts

