# Fake News Detection System Using Machine Learning

## Project Overview

The Fake News Detection System is a Machine Learning and Natural Language Processing (NLP) based web application that predicts whether a news article is REAL or FAKE.

The system preprocesses news text, converts it into numerical vectors using TF-IDF Vectorization, and classifies the news using a Logistic Regression Machine Learning model.

The project is developed using Python, Scikit-learn, and Streamlit.

---

## Features

* Fake and Real News Classification
* NLP Text Preprocessing
* TF-IDF Vectorization
* Machine Learning Prediction
* Confidence Score Display
* News Summary Display
* Interactive Streamlit Web Application

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Natural Language Processing (NLP)

---

## Project Structure

```bash
fake-news-detection-system/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
```

---

## Dataset

Dataset used:

* Fake.csv
* True.csv

Large files are not uploaded to GitHub due to file size limitations.

Dataset Source:
Fake and Real News Dataset:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Step 1: Train the Model

```bash
python train.py
```

### Step 2: Run Streamlit Application

```bash
streamlit run app.py
```

---

## Output

The application predicts:

* REAL NEWS
* FAKE NEWS

along with:

* Confidence Score
* News Summary

---

## Future Enhancements

* Multi-language Support
* AI-based News Explanation
* Sentiment Analysis
* Fake News Source Detection
* Deep Learning Models

---

## Author

Kanikaram Vyshnavi

---

## License

This project is developed for educational and learning purposes.

