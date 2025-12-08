import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import classification_report

print("Loading data...")
df = pd.read_csv("final_project\\preprocessed_edos_labelled_data.csv")  # change path if needed

encoder = LabelEncoder()
y = encoder.fit_transform(df['label']) # 0 for not sexist, 1 for sexist

# mask data
train_mask = df['split'] == 'train'
test_mask = df['split'] == 'test'

# get tweet lists
X_train_tweets = df.loc[train_mask, 'text']
X_test_tweets = df.loc[test_mask, 'text']

# get labels
y_train = y[train_mask]
y_test = y[test_mask]

# tf-idf vectorization
print("Encoding text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), use_idf=True, sublinear_tf=True)

X_train = vectorizer.fit_transform(X_train_tweets)
X_test = vectorizer.transform(X_test_tweets)

# Support Vector Machine model
classifier = LinearSVC(max_iter=1000, C=.1, loss='hinge')
tuned_model = TunedThresholdClassifierCV(estimator=classifier, scoring='f1_weighted')

tuned_model.fit(X_train, y_train)
print(f"Best Threshold Found: {tuned_model.best_threshold_}")
y_pred = tuned_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=encoder.classes_))