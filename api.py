import pandas as pd
from fastapi import FastAPI, Query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import os

# --- Initialization ---
app = FastAPI(
    title="Medium Article Search API",
    description="API to search for top clapped articles based on text/keywords from a scraped dataset."
)

# Global variables for data and vectorizer
df = None
tfidf_vectorizer = None
tfidf_matrix = None

def load_data_and_prepare_model():
    """Loads the data and prepares the TF-IDF model for search."""
    global df, tfidf_vectorizer, tfidf_matrix
    
    data_file = "scrapping_results.csv"
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        return False

    df = pd.read_csv(data_file)
    
    # Combine Title, Subtitle, Text, and Keywords for the search corpus
    df['search_corpus'] = df['Title'].fillna('') + ' ' + \
                          df['Subtitle'].fillna('') + ' ' + \
                          df['Text'].fillna('') + ' ' + \
                          df['Keywords'].fillna('')
    
    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['search_corpus'])
    
    # Convert Claps to numeric, coercing errors to 0
    df['Claps'] = pd.to_numeric(df['Claps'], errors='coerce').fillna(0).astype(int)
    
    print("Data loaded and TF-IDF model prepared.")
    return True

# Load data on startup
load_data_and_prepare_model()

# --- API Endpoints ---

@app.get("/search_articles")
def search_articles(query: str = Query(..., description="Text or keywords to search for."), top_n: int = Query(10, ge=1, le=100)):
    """
    Searches the scraped articles for the most relevant and top-clapped results.
    
    The search works by:
    1. Calculating the cosine similarity between the query and all articles (based on TF-IDF).
    2. Combining the similarity score with the 'Claps' count to rank the results.
    3. Returning the top N articles.
    """
    if df is None:
        return {"error": "Data not loaded. Check if scrapping_results.csv exists."}

    # 1. Vectorize the query
    query_vec = tfidf_vectorizer.transform([query])
    
    # 2. Calculate cosine similarity
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()
    
    # 3. Combine similarity with Claps for a final score
    # Normalize claps (simple min-max normalization)
    min_claps = df['Claps'].min()
    max_claps = df['Claps'].max()
    
    if max_claps > min_claps:
        normalized_claps = (df['Claps'] - min_claps) / (max_claps - min_claps)
    else:
        normalized_claps = pd.Series([0.5] * len(df)) # Default to neutral if all claps are the same

    # Weighted score: 70% from similarity, 30% from claps (can be tuned)
    # This ensures both relevance and popularity are considered.
    final_score = (0.7 * cosine_similarities) + (0.3 * normalized_claps)
    
    # 4. Get the indices of the top-scoring articles
    top_indices = final_score.argsort()[-top_n:][::-1]
    
    # 5. Format the results
    results = []
    for i in top_indices:
        results.append({
            "Title": df.iloc[i]['Title'],
            "URL": df.iloc[i]['URL'],
            "Claps": int(df.iloc[i]['Claps']),
            "Relevance_Score": round(final_score[i], 4)
        })
        
    return results

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "data_loaded": df is not None, "article_count": len(df) if df is not None else 0}

if __name__ == "__main__":
    # This block is for local testing, but for deployment we will use uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
