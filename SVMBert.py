import pandas as pd
from sklearn import svm
from sentence_transformers import SentenceTransformer, models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

print("Loading data...")
df = pd.read_csv("preprocessed_edos_labelled_data.csv")  # Change path if needed

encoder = LabelEncoder()
y = encoder.fit_transform(df['label']) # 0 for not sexist, 1 for sexist

# Load BERT embedder
print("\nLoading bert model...")
bert = models.Transformer('bert-base-uncased', max_seq_length=128)   # 128 tokens otherwise model might not be as accurate, doc said model trained on 128
pooling_model = models.Pooling(bert.get_word_embedding_dimension())  # take the average of the word vectors
embedder = SentenceTransformer(modules=[bert, pooling_model])        # combine bert and the pooler into one embedder

print("Generating embeddings...")
X = embedder.encode(df['text'].tolist())

train_tweets = df['split'] == 'train'
test_tweets = df['split'] == 'test' 

# Apply masks to extract train and test tweets
X_train = X[train_tweets]
X_test = X[test_tweets]

encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])

y_train = y[train_tweets]
y_test = y[test_tweets]

weights = {0:5288/(5288-1565), 1:5288/1565}

classifier = svm.SVC(kernel='linear', class_weight=weights)

print("\nTraining classifier...")
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))