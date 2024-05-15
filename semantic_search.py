import requests
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import faiss
import re

# Suppressing warnings
import warnings
warnings.filterwarnings('ignore')

# Load the Universal Sentence Encoder's TF Hub module
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuations and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fetch data from the public API
url = "https://list.ly/api/v4/meta?url=http://google.com"
response = requests.get(url)
data = response.json()

# Extracting items from the API response
items = [item['title'] for item in data['meta']['listItems']]

# Preprocess each item
processed_items = [preprocess_text(item) for item in items]

# Generate embeddings for each item
X_use = np.vstack([embed([item]).numpy() for item in processed_items])

# Create FAISS index
dimension = X_use.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(X_use)

# Function to perform search
def search(query_text, k=10):
    # Preprocess the query text
    preprocessed_query = preprocess_text(query_text)
    # Generate the query vector
    query_vector = embed([preprocessed_query]).numpy()
    # Perform the search
    distances, indices = index.search(query_vector.astype('float32'), k)
    return distances, indices

# Example Query
query_text = "search query"
distances, indices = search(query_text)

# Display the results
for i, idx in enumerate(indices[0]):
    print(f"Rank {i+1}: (Distance: {distances[0][i]})\n{items[idx]}\n")
