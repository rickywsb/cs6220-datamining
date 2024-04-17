import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import re
import numpy as np
from joblib import Parallel, delayed

# Load dataset
df = pd.read_csv('Duolingo_dataset_scaled.csv')


# Function to clean and tokenize lexeme strings
def clean_and_tokenize(lexeme):
    lexeme = re.sub(r'[<>]', ' ', lexeme)  # Remove angle brackets
    tokens = word_tokenize(lexeme)
    return tokens

# Parallel processing using Joblib
def process_chunk(chunk):
    chunk['tokens'] = chunk['lexeme_string'].apply(clean_and_tokenize)
    return chunk

# Read and process data in chunks
chunksize = 5000  # Adjust this based on your memory constraints
chunks = pd.read_csv('Duolingo_dataset_scaled.csv', chunksize=chunksize)
processed_chunks = Parallel(n_jobs=-1)(delayed(process_chunk)(chunk) for chunk in chunks)
df = pd.concat(processed_chunks)

# Now that we have tokens, we can train the Word2Vec model
model = Word2Vec(sentences=df['tokens'].tolist(), vector_size=50, window=5, min_count=1, workers=4)

# Function to vectorize lexeme strings using the trained Word2Vec model
def vectorize(tokens):
    vector = np.mean([model.wv[token] for token in tokens if token in model.wv], axis=0)
    if np.isnan(vector).any():
        return np.zeros(model.vector_size)
    return vector

df['vector'] = df['tokens'].apply(vectorize)

# Convert vectors from lists to separate columns and concatenate
vector_df = pd.DataFrame(df['vector'].tolist(), columns=[f'vec_{i}' for i in range(model.vector_size)])
df = pd.concat([df, vector_df], axis=1)
df.drop(['lexeme_string', 'tokens', 'vector'], axis=1, inplace=True)

# Save to CSV
df.to_csv('Duolingo_dataset_nlp_processed.csv', index=False)

print("DataFrame processed with NLP techniques and saved successfully.")

