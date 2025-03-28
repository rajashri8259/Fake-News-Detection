# Fake News Detection with Python

This project implements a machine learning model to detect fake news using Python. The model uses TF-IDF vectorization and a Passive Aggressive Classifier to classify news articles as either REAL or FAKE.

## Features

- TF-IDF vectorization for text processing
- Passive Aggressive Classifier for classification
- Confusion matrix visualization
- Support for custom news article prediction
- High accuracy (typically >90%)

## Prerequisites

- Python 3.7 or higher
- Required Python packages (install using `pip install -r requirements.txt`)

## Installation

1. Clone this repository or download the files
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download the dataset:
   - Download the news.csv dataset from [here](https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view)
   - Place it in the same directory as the script

## Usage

1. Run the main script:
   ```
   python fake_news_detector.py
   ```

2. The script will:
   - Load and prepare the dataset
   - Train the model
   - Show the model's accuracy
   - Display a confusion matrix
   - Test with a sample news article

3. To predict your own news article:
   - Modify the `sample_news` variable in the script
   - Run the script again

## How it Works

1. **Data Preparation**:
   - The dataset is split into training and testing sets
   - Text is converted to TF-IDF features

2. **Model Training**:
   - Uses TF-IDF vectorization to convert text to numerical features
   - Trains a Passive Aggressive Classifier on the features

3. **Prediction**:
   - Converts input text to TF-IDF features
   - Uses the trained model to predict if the news is real or fake

## Model Performance

The model typically achieves:
- Accuracy: >90%
- Confusion matrix shows true positives, true negatives, false positives, and false negatives

## Contributing

Feel free to submit issues and enhancement requests! 