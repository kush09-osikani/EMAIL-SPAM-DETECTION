Below is the Readme for the Email Spam Detection Project using Machine learning

# EmailSpamDetection — README

This folder contains a small example script `EmailSpamDetection.py` that trains a simple Naive Bayes text classifier to detect spam vs. non-spam (ham) messages and exposes a small Streamlit UI for manual testing.

Contents
- `EmailSpamDetection.py` — training and a tiny Streamlit app to validate single messages
- `spam.csv` — dataset used by the script

Dependencies
- Python 3.8+ (3.11 is fine)
- pandas
- numpy
- scikit-learn
- streamlit

Install with pip:
```powershell
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn streamlit
```

What the script does
- Loads `spam.csv` (hard-coded path in the script). The file is expected to have at least two columns named `Category` and `Message`.
- Performs basic preprocessing: drops duplicate rows, reports missing values, and remaps the label values `spam`/`ham` to `Not Spam`/`Spam` (note: this mapping in the script is inverted — see "Notes/Warnings" below).
- Splits the dataset into train/test (80/20) using scikit-learn's `train_test_split`.
- Vectorizes messages with `CountVectorizer(stop_words='english')` and trains a `MultinomialNB` classifier.
- Prints accuracy on the test set and demonstrates a single prediction.
- Exposes a Streamlit UI where you can type a message and click `Validate` to see the model prediction.

How to run locally
1) From PowerShell (or a prompt using the same Python environment):
```powershell
python "C:\Users\ICL512\Desktop\VS CODE WORKSPACE\MACHINE LEARNING\EmailSpamDetection.py"
```

2) To run the interactive Streamlit app (recommended for the UI):
```powershell
cd "C:\Users\ICL512\Desktop\VS CODE WORKSPACE\MACHINE LEARNING"
streamlit run EmailSpamDetection.py
```
Then open the URL printed by Streamlit (usually http://localhost:8501) in your browser.

Notes / Warnings
- Hard-coded dataset path: The script currently reads the CSV using an absolute path. For portability change this to a relative path (e.g., `pd.read_csv('spam.csv')`) or add an argparse option.
- Label mapping: The script runs a mapping that appears inverted (`spam -> Not Spam`, `ham -> Spam`). This will flip the meaning of predictions. Change to `['spam','ham'] -> ['Spam','Not Spam']` if you want conventional labels.
- Model persistence: The script trains a `MultinomialNB` model in-memory and does not save it. If you want to reuse the trained model without retraining, add model persistence using `joblib` or `pickle`.

Suggested improvements
- Fix the label mapping inversion.
- Add command-line flags (argparse) for dataset path, test split ratio, and random seed.
- Persist the trained model to disk (`joblib.dump(model, 'model.joblib')`) and add a `--predict-only` mode that loads the saved model.
- Replace `CountVectorizer` with `TfidfVectorizer` and try basic hyperparameter tuning (grid search) to improve accuracy.
- Add proper evaluation metrics: confusion matrix, precision, recall and F1-score (use `sklearn.metrics.classification_report`).
- Use a pipeline (`sklearn.pipeline.Pipeline`) to combine vectorizer and classifier and make code cleaner and safer for production.

Example improvements (code snippets)
- Save trained model with joblib:
```python
from joblib import dump, load
dump(model, 'spam_nb.joblib')
# later: model = load('spam_nb.joblib')
```
- Use a pipeline and TF-IDF:
```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipe = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
pipe.fit(x_train, y_train)
print(pipe.score(x_test, y_test))
```

Troubleshooting
- If Streamlit doesn't start, ensure the `streamlit` package is installed in the same Python environment the `streamlit` command uses. Validate with `python -m streamlit hello`.
- If `ModuleNotFoundError` appears for scikit-learn or pandas, install them with `pip` into the same interpreter you run the script with.

Contact / Next steps
If you'd like, I can:
- Fix the label mapping and add model persistence.
- Convert the script into a Streamlit app with a saved model and clean UI.
- Add a CLI and refactor the training/evaluation into functions with unit tests.

Tell me which improvements you'd like and I'll implement them.
