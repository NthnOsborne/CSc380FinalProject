import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stop_words(inp):
    words = nltk.word_tokenize(str(inp))
    no_stop_words = []
    for word in words:
        if word not in stop_words:
            no_stop_words.append(word)
    return ' '.join(no_stop_words)

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = nltk.word_tokenize(str(text))
    lemmas = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(lemmas)

data = pd.read_csv("edos_labelled_data.csv")

# make the columns mirror the original data given
# keep only the columns you want and rename label_sexist -> label
data = data[['rewire_id', 'text', 'label_sexist', 'split']]
data = data.rename(columns={'label_sexist': 'label'})


# apply the same preprocessing
# Lowercase
data['text'] = data['text'].str.lower()
# Remove URLs
data['text'] = data['text'].str.replace(r'http\S+|www\S+', '', regex=True)
# Remove HTML tags
data['text'] = data['text'].str.replace(r'<.*?>', '', regex=True)
# Remove remaining non-alphanumeric characters (emojis, punctuation)
data['text'] = data['text'].str.replace(r'[^a-zA-Z0-9 ]', '', regex=True)
# Remove stop words
data['text'] = data['text'].apply(remove_stop_words)
# Lemmatize
data['text'] = data['text'].apply(lemmatize_text)

data.to_csv("preprocessed_edos_labelled_data.csv", index=False)