import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load pre-trained components
model = joblib.load("src/resume_classification_model.pkl")
tokenizer = joblib.load("src/resume_label_encoder.pkl")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def predict_resume_category(resume_texts):
    """
    Given a list or Series of resume texts, return their predicted categories.
    """
    if isinstance(resume_texts, str):
        resume_texts = [resume_texts]

    # Generate embeddings
    embeddings = sbert_model.encode(resume_texts)

    # Predict class labels
    predicted_labels = model.predict(embeddings)

    # Decode labels to category names
    predicted_categories = tokenizer.inverse_transform(predicted_labels)

    return predicted_categories
