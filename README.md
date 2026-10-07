Fake News Detection using Logistic Regression

An end-to-end Python pipeline for detecting fake news using Natural Language Processing (NLP) and Logistic Regression. This project reads news datasets, preprocesses textual data, converts text into numerical features using TF-IDF Vectorization, trains a Logistic Regression model, and exports the trained artifacts (.jb files) for deployment.

📌 Project Overview:

This script performs the following workflow:

Data Ingestion: Loads Fake.csv and True.csv datasets.

Labeling & Concatenation: Assigns 0 for fake news and 1 for true news, then combines the datasets.

Text Preprocessing: Cleans raw text by converting to lowercase and removing URLs, HTML tags, punctuation, special characters, and newlines.

Feature Extraction: Transforms news text into TF-IDF vectors using TfidfVectorizer.

Model Training: Trains a LogisticRegression classifier on a 75/25 train-test split.

Evaluation: Prints dataset dimensions and a complete classification report (Precision, Recall, F1-Score).

Model Serialization: Saves the trained vectorizer (vectorizer.jb) and Logistic Regression model (lr_model.jb) using joblib.

📁 Required Dataset Structure(Kaggle):

Place your dataset files in the root folder alongside the script:

Fake.csv: Dataset containing fake news articles.

True.csv: Dataset containing legitimate news articles.

Both CSV files should contain a text column with the article content.

🛠️ Requirements & Installation:

Make sure you have Python 3.x installed along with the following libraries:

pip install pandas scikit-learn joblib

🔍 Model Pipeline Summary:

Train/Test Split: 75% Training, 25% Testing (random_state=42).

Vectorization: TfidfVectorizer (scikit-learn).

Classifier: LogisticRegression (scikit-learn).

Evaluation Metrics: Classification Report via sklearn.metrics
Precision: Out of all articles predicted as fake (or real), how many were actually fake (or real)?
Recall: Out of all actual fake (or real) articles, how many did the model successfully identify?
F1-Score: The harmonic mean of Precision and Recall, summarizing model performance per class.

🚀 Execution:

Run the training script using Python:

python train.py

After execution, the following serialized model files will be created in your working directory:

vectorizer.jb: Saved TF-IDF Vectorizer instance.

lr_model.jb: Saved Logistic Regression trained model.

Final output:

streamlit run app.py

How the Data is Trained to Predict "Fake" vs "Real"

Labeling the Data:

Two datasets are used: Fake.csv (labeled as 0) and True.csv (labeled as 1).

They are merged into a single dataset with text articles and their corresponding class labels.


Text Preprocessing (clean_text):

Raw text contains noise like URLs, HTML tags, special symbols, punctuation, and extra whitespace.
The clean_text function strips out these irrelevant characters and converts all text to lowercase, normalizing the text so the model focuses purely on meaningful words.

TF-IDF Vectorization:

TF-IDF (Term Frequency-Inverse Document Frequency) converts words into numerical features based on,

Term Frequency (TF): How often a word appears in a specific article.

Inverse Document Frequency (IDF): How unique or common that word is across all articles.

Logistic Regression Classifier:

A Logistic Regression model learns the statistical relationship between these TF-IDF word weights and the class labels (0 or 1).
It calculates the probability of an article belonging to class 0 (Fake) or class 1 (Real)

Saving the Artifacts:

Both the fitted vectorizer and trained lr model are saved to disk using joblib (vectorizer.jb and lr_model.jb) so they can be re-loaded into any web interface without re-training every time.

Streamlit:

Web UI for Real-Time Inference: Streamlit turns your Python script into an interactive web application with minimal code.
Loading Saved Models: Using joblib.load(), Streamlit loads the saved vectorizer.jb and lr_model.jb instantly when the app opens.

User Input & Prediction Flow:

A user pastes a news article or headline into a text area (st.text_area).
Streamlit applies the exact same clean_text function to the input.
The cleaned text is transformed using the loaded vectorizer.
The loaded lr model runs .predict() on the vectorized input.
The UI displays an alert box (e.g., st.error("⚠️ Fake News Detected") for class 0 or st.success("✅ Real News Article") for class 1).


