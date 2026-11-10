import streamlit as st
import numpy as np
import re
import io

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


SKILL_ALIASES = {
    "Python": ["python"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "JavaScript": ["javascript", "js"],
    "SQL": ["sql"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "NLP": ["natural language processing", "nlp"],
    "Computer Vision": ["computer vision"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Data Structures": ["data structures", "dsa"],
    "Algorithms": ["algorithms"],
    "Git": ["git"],
    "GitHub": ["github"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "Linux": ["linux"],
    "Flask": ["flask"],
    "Django": ["django"],
    "FastAPI": ["fastapi"],
    "Streamlit": ["streamlit"],
    "React": ["react"],
    "Node.js": ["node.js", "nodejs"],
    "HTML": ["html"],
    "CSS": ["css"],
    "MongoDB": ["mongodb"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "REST API": ["rest api", "restful api"],
    "Blockchain": ["blockchain"],
    "Solidity": ["solidity"],
    "Excel": ["excel"],
    "Power BI": ["power bi"],
}


def extract_resume_text(uploaded_file):

    pdf = PdfReader(
        io.BytesIO(uploaded_file.getvalue())
    )

    if pdf.is_encrypted:
        raise ValueError("Password-protected PDFs are not supported.")

    text = ""

    for page in pdf.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"

    return text.strip()


def split_into_chunks(text, max_words=150):

    words = text.split()

    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks


def create_embedding(text, model):

    chunks = split_into_chunks(text)

    if not chunks:
        raise ValueError("No text available to analyze.")

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Average the embeddings of all chunks
    document_embedding = np.mean(
        embeddings,
        axis=0,
        keepdims=True
    )

    # Normalize the final document embedding
    norm = np.linalg.norm(
        document_embedding,
        axis=1,
        keepdims=True
    )

    document_embedding = document_embedding / np.maximum(norm, 1e-12)

    return document_embedding


def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill, aliases in SKILL_ALIASES.items():

        for alias in aliases:

            pattern = (
                r"(?<!\w)"
                + re.escape(alias.lower())
                + r"(?!\w)"
            )

            if re.search(pattern, text):

                found_skills.append(skill)
                break

    return sorted(found_skills)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def load_model():

    model = SentenceTransformer(MODEL_NAME)

    return model


st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide"
)

st.title("AI Resume and Job Description Matcher")

st.write(
    "Upload your resume and compare it with a job description "
    "using Natural Language Processing."
)

st.info(
    "Upload a text-based PDF resume. Avoid using confidential "
    "personal documents in the public demo."
)

uploaded_file = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste the Job Description",
    height=250,
    placeholder="Paste the complete job description here..."
)

analyze = st.button(
    "Analyze Resume",
    type="primary"
)

if analyze:

    if uploaded_file is None:
        st.error("Please upload your resume.")

    elif not job_description.strip():
        st.error("Please enter a job description.")

    elif uploaded_file.size > 5 * 1024 * 1024:
        st.error("Please upload a PDF smaller than 5 MB.")

    else:

        try:
            resume_text = extract_resume_text(uploaded_file)

        except Exception as e:
            st.error(f"Could not read the PDF: {e}")
            st.stop()

        if not resume_text:
            st.error("No readable text found in the PDF.")
            st.info(
                "Scanned or image-only PDFs are not supported "
                "in this version."
            )
            st.stop()

        with st.spinner("Analyzing resume..."):

            model = load_model()

            # Generate embeddings
            resume_embedding = create_embedding(
                resume_text, model
            )

            job_embedding = create_embedding(
                job_description, model
            )

            # Calculate semantic similarity
            similarity = cosine_similarity(
                resume_embedding,
                job_embedding
            )[0][0]

            # Extract skills
            resume_skills = set(
                extract_skills(resume_text)
            )

            job_skills = set(
                extract_skills(job_description)
            )

            matched_skills = sorted(
                resume_skills & job_skills
            )

            missing_skills = sorted(
                job_skills - resume_skills
            )

            additional_skills = sorted(
                resume_skills - job_skills
            )

            # Calculate skill coverage
            if job_skills:
                skill_coverage = (
                    len(matched_skills) /
                    len(job_skills)
                ) * 100
            else:
                skill_coverage = 0

        # Display results
        st.divider()

        st.header("Resume Analysis Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Semantic Similarity",
            f"{similarity:.3f}"
        )

        col2.metric(
            "Matching Skills",
            len(matched_skills)
        )

        col3.metric(
            "Skill Coverage",
            f"{skill_coverage:.1f}%"
            if job_skills else "N/A"
        )

        st.caption(
            "Semantic similarity is a model score, not a "
            "probability or official ATS rating. Skill coverage "
            "uses only the predefined skill dictionary."
        )

        st.divider()

        st.subheader("Matching Skills")

        if matched_skills:
            st.success(", ".join(matched_skills))
        else:
            st.info("No matching skills detected.")

        st.subheader("Potential Missing Skills")

        if missing_skills:
            st.warning(", ".join(missing_skills))
            st.write(
                "Consider these skills only if they genuinely "
                "match your experience and career goals."
            )
        else:
            st.success("No missing skills detected by our dictionary.")

        st.subheader("Additional Resume Skills")

        if additional_skills:
            st.info(", ".join(additional_skills))
        else:
            st.write("No additional skills detected.")

        st.progress(
            int(skill_coverage)
        ) if job_skills else None

        with st.expander("View Extracted Resume Text"):
            st.text(resume_text)

