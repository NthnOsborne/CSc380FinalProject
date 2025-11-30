import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
def remove_stop_words(inp):
    words = nltk.word_tokenize(inp)
    no_stop_words = []
    for word in words:
        if word not in stop_words:
            no_stop_words.append(word)
    return ' '.join(no_stop_words)

lemmatizer = WordNetLemmatizer()
def lemmatize_text(inp):
    words = nltk.word_tokenize(inp)
    lem_words = []
    for word in words:
        lem_words.append(lemmatizer.lemmatize(word))
    return ' '.join(lem_words)

data = pd.read_csv('edos_labelled_data.csv')

# Lowercase
data['text'] = data['text'].str.lower()
# URLs
data['text'] = data['text'].str.replace(r'http\S+|www\S+', '', regex=True)
# HTML tags
data['text'] = data['text'].str.replace(r'<.*?>', '', regex=True)
# Any remaining non alphanumerical characters(emojis, puncuation)
data['text'] = data['text'].str.replace(r'[^a-zA-Z0-9 ]', '', regex=True)
# Remove stop words
data['text'] = data['text'].apply(remove_stop_words)
# Lemmatize
data['text'] = data['text'].apply(lemmatize_text)

data.to_csv('preprocessed_edos_labelled_data.csv', index=False)