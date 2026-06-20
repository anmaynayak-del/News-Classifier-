# News Category Classifier

A Streamlit web app that classifies headlines or short news articles into:

- Sports
- Business
- Technology
- Politics

The app uses TF-IDF vectorization with a scikit-learn Logistic Regression classifier.

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit app:

```bash
streamlit run app.py
```

## Quick Model Check

You can test the classifier logic from the terminal:

```bash
python classifier.py
```
