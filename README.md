
# AI Resume and Job Description Matcher

An NLP-based application that compares resumes
with job descriptions using semantic similarity
and skill extraction.

## Features

- Upload PDF resumes
- Extract resume text automatically
- Compare resumes with job descriptions
- Calculate semantic similarity
- Identify matching skills
- Detect potential missing skills
- Display an interactive analysis dashboard

## Technologies Used

- Python
- Streamlit
- Sentence Transformers
- Scikit-learn
- NumPy
- PyPDF

## NLP Model

sentence-transformers/all-MiniLM-L6-v2

## How It Works

1. Extract text from the uploaded resume.
2. Convert resume and job description into embeddings.
3. Calculate cosine similarity.
4. Extract technical skills using a predefined dictionary.
5. Display matching and potentially missing skills.

## Installation

Install dependencies:

python -m pip install -r requirements.txt

Run the application:

python -m streamlit run app.py

## Limitations

- Supports text-based PDF files.
- Scanned PDFs require OCR.
- Skill extraction uses a predefined dictionary.
- Similarity is not an official ATS or hiring score.

## Author

sahwaz khan

## GitHub

https://github.com/kira-445/AI-Resume-Job-Matcher

## Live Demo

https://ai-resume-job-matcher-nrnpmacm9sj7vscgvfwp52.streamlit.app/
