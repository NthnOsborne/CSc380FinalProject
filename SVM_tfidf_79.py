from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd

# 1. Load data
df = pd.read_csv("preprocessed_edos_labelled_data.csv")

# Use the given split column
train_mask = df["split"] == "train"
test_mask = df["split"] == "test"

X_train = df.loc[train_mask, "text"]
y_train = df.loc[train_mask, "label"]
X_test  = df.loc[test_mask, "text"]
y_test  = df.loc[test_mask, "label"]

# 2. Build pipeline: word-level TF-IDF + Linear SVM
tfidf_word = TfidfVectorizer(
    analyzer="word",
    sublinear_tf=True          # log-scale term frequencies
)

pipe = Pipeline([
    ("tfidf", tfidf_word),
    ("clf", LinearSVC())       # linear SVM for sparse text
])

# 3. Hyperparameter grid (kept small so it actually finishes)
param_grid = {
    # Text representation
    "tfidf__ngram_range": [(1, 2)],          # unigrams + bigrams
    "tfidf__max_features": [10000, None],    # vocab size
    "tfidf__min_df": [1, 2, 3],              # drop super-rare words
    "tfidf__stop_words": [None],             # keep pronouns etc. for sexism

    # SVM regularization
    "clf__C": [0.5, 1.0, 2.0, 5.0],
    "clf__class_weight": ["balanced"],       # handle class imbalance
}

# 4. Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_weighted",        
    n_jobs=-1,
    verbose=2
)

print("Running grid search...")
grid.fit(X_train, y_train)

print("\nBest params:", grid.best_params_)
print("Best CV F1-weighted:", grid.best_score_)

# 5. Evaluate on held-out test set
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nTest classification report (using best model):")
print(classification_report(y_test, y_pred))

macro_f1 = f1_score(y_test, y_pred, average="macro")
print("Test F1-macro:", macro_f1)
