import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

# Load the data
with open('Food_names.json', 'r') as f:
    food_data = json.load(f)
food_names = food_data["foods"]

char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 6), lowercase=True)
X = char_vectorizer.fit_transform(food_names)
char_nn = NearestNeighbors(n_neighbors=10, metric='cosine')
char_nn.fit(X)

word_vectorizer = TfidfVectorizer(lowercase=True)
X = word_vectorizer.fit_transform(food_names)
word_nn = NearestNeighbors(n_neighbors=10, metric='cosine')
word_nn.fit(X)

# save the models

joblib.dump(char_vectorizer, 'char_vectorizer.pkl')
joblib.dump(char_nn, 'char_nn.pkl')
joblib.dump(word_vectorizer, 'word_vectorizer.pkl')
joblib.dump(word_nn, 'word_nn.pkl')
