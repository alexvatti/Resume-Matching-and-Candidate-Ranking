import numpy as np
import pandas as pd
import re

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