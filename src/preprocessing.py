import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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
