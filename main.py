import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
# Import the TF-IDF vectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
# sentence-BERT embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Load the English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")

# Function to fix encoding issues in text
def fix_encoding(text):
    return text.encode('latin1').decode('utf-8', errors='ignore')  # or use errors='replace'

# Define text preprocessing function
def preprocess_text(text):
    """
    Preprocess resume text through the following steps:
    1. Load resume data from CSV files.
    2. Clean and standardize text:
       - Convert to lowercase
       - Remove numbers and punctuation
       - Tokenize text
       - Remove stop words
       - Apply lemmatization
    """

    # Step 3: Convert text to lowercase
    text = text.lower()

    # Step 4: Remove punctuation and numbers
    text = re.sub(r'[^\w\s]|[\d]', '', text)

    # Step 5: Tokenize text into words
    tokens = word_tokenize(text)

    # Step 6 & 7: Remove stop words and apply lemmatization
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Join tokens back into a single string
    return ' '.join(cleaned_tokens)

# Function to extract skill, education, and work experience-related lines from resume text
def extract_skill_edu_exp(text):
    """Function to extract skill, education, and work experience-related lines from resume text"""
    # Define a set of skill-related keywords
    skills_keywords = {
        "skills", "skill", "technical skills", "soft skills", "tools", "technologies", "technology",
        "frameworks", "libraries", "platforms", "languages", "certifications", "methodologies",
        "programming", "databases", "cloud", "devops", "analytics", "testing tools", "networking"
    }

    # Define a comprehensive set of education-related keywords
    education_keywords = {
        # General education terms
        "bachelor", "bachelors", "master", "masters", "phd", "degree", "degrees", "university", "college",
        "graduate", "postgraduate", "undergraduate", "school", "education", "certification", "certifications", "diploma",

        # Common degree/program abbreviations
        "b.tech", "be", "bsc", "bca", "bba", "ba", "mba", "mca", "m.tech", "me", "msc", "ms", "pgdm", "pg", "ug", "llb", 
        "llm",

        # Data/Tech-specific
        "data science", "machine learning", "artificial intelligence", "computer science", "cs",
        "information technology", "it", "software engineering", "cybersecurity", "network security",
        "devops", "cloud computing", "aws", "azure", "gcp", "database", "big data", "hadoop", "etl",
        "python", "java", "dotnet", ".net", "blockchain", "web development", "web designing", "ui ux",
        "software testing", "automation testing", "qa", "full stack", "frontend", "backend",

        # Business/Management
        "business administration", "business analytics", "operations management", "operations",
        "human resources", "hr", "marketing", "finance", "accounting", "commerce", "economics",
        "entrepreneurship", "strategy", "organizational behavior", "project management", "pmp",

        # Engineering branches
        "mechanical engineering", "civil engineering", "electrical engineering", "electronics",
        "electronics and communication", "ece", "instrumentation", "engineering",

        # Legal/Advocate
        "law", "legal studies", "advocate", "llb", "llm", "jurisprudence", "judicial", "legal",

        # Arts & Health
        "liberal arts", "fine arts", "performing arts", "visual arts", "design", "health sciences",
        "healthcare", "nursing", "medicine", "fitness", "physical education", "nutrition",

        # Additional soft skill / professional-related
        "communication", "leadership", "pmo", "analyst", "consultant", "trainer", "coach"
    }

    # Define work-related keywords
    work_keywords = {
        "experience", "exprience", "worked", "employed", "company", "organization", "intern", "internship"
    }

    # Compile regex patterns for matching categories
    skills_patten = re.compile(r'\b(' + '|'.join(skills_keywords) + r')\b', re.IGNORECASE)
    edu_pattern = re.compile(r'\b(' + '|'.join(education_keywords) + r')\b', re.IGNORECASE)
    work_pattern = re.compile(r'\b(' + '|'.join(work_keywords) + r')\b', re.IGNORECASE)
    # Lists to store matched lines
    skills = []
    education = []
    experience = []

    # Split text into lines to process individually
    lines = text.split('\n')

    for line in lines:
        sentence = line.strip()
        lower = sentence.lower()

        # Match and collect skill-related lines
        if skills_patten.search(lower):
            skills.append(sentence)

        # Match and collect education-related lines
        if edu_pattern.search(lower):
            education.append(sentence)

        # Match and collect work experience-related lines
        if work_pattern.search(lower):
            experience.append(sentence)

    # Return results as a Pandas Series (suitable for applying to DataFrames)
    return pd.Series({
        "Skills": ",".join(skills),
        "Education": ",".join(education),
        "Experience": ",".join(experience)
    })

# Define a function to convert a list of resume texts into TF-IDF feature vectors
def vectorize_resumes(resume_list):
    """
    Takes a list of resumes (text strings) and returns a DataFrame
    of TF-IDF features.
    TF-IDF reflects how important a word is to a document in a collection (corpus).
    """
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

def classify_resumes(resumes, labels, new_resume=None):
    """
    Preprocesses resumes, trains a classifier, and optionally classifies a new resume.
    
    Args:
        resumes (list): List of resume texts.
        labels (list): List of job category labels.
        new_resume (str, optional): A new resume text to classify. Default is None.
        
    Returns:
        str: If `new_resume` is provided, returns the predicted job category for the new resume.
        None: Otherwise, trains and evaluates the model and prints classification report.
    """
        # Encode resumes using Sentence-BERT
    X = embed_resumes_with_sbert(resumes)
    
    # Encode job labels into numeric values
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Logistic Regression model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    # If a new resume is provided, classify it
    if new_resume:
        # Encode the new resume text
        embedding = embed_resumes_with_sbert([new_resume])
        
        # Predict the job category
        pred = clf.predict(embedding)
        
        # Convert the numerical prediction back to the original label
        return le.inverse_transform(pred)[0]
    else:
        # No new resume, just return None
        return None

def rank_resumes_by_similarity(job_description, resumes):
    """
    Ranks resumes based on their semantic similarity to a given job description.

    Args:
        job_description (str): The job description text.
        resumes (list): A list of resume texts.

    Returns:
        list: List of tuples (index, similarity_score) sorted by similarity_score in descending order.
    """

    # Generate embedding for job description
    job_embedding = embed_resumes_with_sbert([job_description]).values[0].reshape(1, -1)

    # Generate embeddings for all resumes
    resume_embeddings = embed_resumes_with_sbert(resumes).values  # Now a proper 2D array

    # Calculate cosine similarity for each resume
    similarities = cosine_similarity(job_embedding, resume_embeddings)[0]

    # Rank resumes with index
    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    return ranked


if __name__ == "__main__":
    # Step 1: Load the resume data from a CSV file
    csv_file = 'UpdatedResumeDataSet.csv'
    column_name = 'Resume'
    df = pd.read_csv(csv_file, encoding='utf-8')

    # Step 2: Fix any encoding issues in the Resume column
    df['Resume'] = df['Resume'].apply(fix_encoding)

    # Step 3: Preprocess the resume text (e.g., lowercase, remove stopwords, clean punctuation)
    df["Resume_processed"] = df["Resume"].astype(str).apply(preprocess_text)

    # Step 4: Extract structured information - Skills, Education, and Experience - from the original resume text
    df[['Skills', 'Education', 'Experience']] = df['Resume'].apply(extract_skill_edu_exp)
    df.to_csv("output.csv",index=False)

    # Step 5: Apply TF-IDF vectorization on the preprocessed resume text
    resumes = df["Resume_processed"].fillna('')  # Ensure there are no NaN values
    tfidf_df = vectorize_resumes(resumes)

    # Display the TF-IDF feature matrix (each row is a resume, each column is a term)
    print(tfidf_df)

    # Step 6: Generate semantic embeddings for the resumes using Sentence-BERT (SBERT)
    sbert_df = embed_resumes_with_sbert(resumes)

    # Display the SBERT embeddings DataFrame (each row is a vector representation of a resume)
    print(sbert_df)


    # Train the model and print the classification report
    classify_resumes(df["Resume"], df["Category"])
    
    print("Resume_processed-------------------")
    resumes = df["Resume_processed"][0:20]
    job_description = """
    Skills:

    Proven experience of 6 + years as an Data Scientist

    Solid experience in data science ML modelling (Classification, Regressions, time series, clustering, pattern recognition) optimization, reinforcement learning, deep learning algorithms, cloud computing architecture and big data management and reporting, NLP and Generative AI
    Knowledge of Python, SQL , Azure ML
    Familiarity with business intelligence tools/ data Visualization tools - Power BI
    Demonstrates an understanding of the IT environment, including enterprise applications like HRMS, ERPs, CRM, etc.

    Educational Qualification:

    Graduate/Post Graduate degree Computer Science, Statistics, Data Science or a

    related field
    """
    ranked_results = rank_resumes_by_similarity(job_description, resumes)

    # Display ranked resumes
    print(f"{'Resume':<8} | {'Index':<5} | {'Score':<7} | {'Rank':<4}")
    print("-" * 36)
    for rank, (idx, score) in enumerate(ranked_results, start=1):
        print(f"{'Resume':<8} | {idx:<5} | {score:<7.4f} | {rank:<4}")

    print("Resume------------------------")
    resumes = df["Resume"][0:20]
    ranked_results = rank_resumes_by_similarity(job_description, resumes)

    # Display ranked resumes
    print(f"{'Resume':<8} | {'Index':<5} | {'Score':<7} | {'Rank':<4}")
    print("-" * 36)
    for rank, (idx, score) in enumerate(ranked_results, start=1):
        print(f"{'Resume':<8} | {idx:<5} | {score:<7.4f} | {rank:<4}")

    df['skill_edu_exp_resume'] = df[['Skills', 'Education', 'Experience']].astype(str).agg(' '.join, axis=1)
    print("skill_edu_exp_resume ----------------------------")
    resumes = df["skill_edu_exp_resume"][0:20]
    ranked_results = rank_resumes_by_similarity(job_description, resumes)

    # Display ranked resumes
    print(f"{'Resume':<8} | {'Index':<5} | {'Score':<7} | {'Rank':<4}")
    print("-" * 36)
    for rank, (idx, score) in enumerate(ranked_results, start=1):
        print(f"{'Resume':<8} | {idx:<5} | {score:<7.4f} | {rank:<4}")