import pandas as pd
import numpy as np
# Import the TF-IDF vectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
# sentence-BERT embeddings
from sentence_transformers import SentenceTransformer


# Define a function to convert a list of resume texts into TF-IDF feature vectors
def vectorize_resumes(resume_list):
    """
    Takes a list of resumes (text strings) and returns a DataFrame
    of TF-IDF features.
    TF-IDF reflects how important a word is to a document in a collection (corpus).
    """
    if not resume_list:
        return []
    # Create an instance of TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit the model on the text data and transform it into a sparse matrix
    tfidf_matrix = vectorizer.fit_transform(resume_list)

    # Get the feature names (i.e., the words/tokens) from the fitted model
    feature_names = vectorizer.get_feature_names_out()

    # Convert the sparse matrix into a dense DataFrame with feature names as columns
    return pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

def embed_resumes_with_sbert(resumes, model_name='all-MiniLM-L6-v2'):
    """
    Takes a Series or list of resumes,
    returns a DataFrame of sentence-BERT embeddings.
    """
    # Load sentence-BERT model
    model = SentenceTransformer(model_name)
    
    # Encode resumes
    embeddings = model.encode(resumes, show_progress_bar=True)
    
    # Create a new DataFrame from the embeddings
    embedding_df = pd.DataFrame(embeddings, index=resumes.index if isinstance(resumes, pd.Series) else None)
    
    return embedding_df    