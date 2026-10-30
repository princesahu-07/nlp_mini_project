# Project 1: Text Cleaner and Keyword Finder (NLP)

## Objective
Analyze 100–200 SMS messages with lowercase conversion, punctuation removal, English stop-word removal, and Porter stemming. Show the 10 most frequent stemmed words as a bar chart, and top 10 keywords ranked by summed TF-IDF scores.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app automatically reads a public mirror of the SMS Spam Collection dataset, or you may upload the Kaggle `spam.csv` (columns `v1`, `v2`), or place `spam.csv` next to `app.py`.

Kaggle dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
Public mirror: https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

Select 100–200 messages; the random sampling seed (42) ensures reproducible results. Download the cleaned CSV using the button.
