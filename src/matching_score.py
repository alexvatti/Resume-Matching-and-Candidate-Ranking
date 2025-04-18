from src.feature_extraction import embed_resumes_with_sbert
from sklearn.metrics.pairwise import cosine_similarity

def rank_resumes_by_similarity(job_description, resumes):
    """
    Ranks resumes based on their semantic similarity to a given job description.

    Args:
        job_description (str): The job description text.
        resumes (list): A list of resume texts.

    Returns:
        list: List of tuples (index, similarity_score) sorted by similarity_score in descending order.
    """
    if not job_description or not resumes:
        raise ValueError("Job description and resumes must be provided.")

    # Generate embedding for job description
    job_embedding = embed_resumes_with_sbert([job_description]).values[0].reshape(1, -1)

    # Generate embeddings for all resumes
    resume_embeddings = embed_resumes_with_sbert(resumes).values  # 2D array

    # Calculate cosine similarity
    similarities = cosine_similarity(job_embedding, resume_embeddings)[0]

    # Rank by similarity
    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    return ranked
