
import streamlit as st
import pandas as pd
import re
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import pipeline

# ---------------------------
# Load FAISS and Verse Dict
# ---------------------------
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("geeta_faiss_index", embeddings)

    df = pd.read_csv("geeta_dataset.csv")
    verse_dict = {(int(r.chapter), int(r.verse)): Document(
        page_content=r.english,
        metadata={
            "chapter": int(r.chapter),
            "verse": int(r.verse),
            "sanskrit": r.sanskrit.strip(),
            "english": r.english.strip(),
            "id": f"{r.chapter}.{r.verse}"
        }
    ) for _, r in df.iterrows()}

    generator = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        device=0
    )
    return vector_db, verse_dict, generator

vector_db, verse_dict, generator = load_resources()


def clean_response(text: str) -> str:
    text = re.sub(r"Please.*", "", text, flags=re.I)
    text = re.sub(r"(🙏\s*){2,}", "🙏", text)
    text = re.sub(r"(May this wisdom guide you[^\n]*){2,}", "May this wisdom guide you. 🙏", text)
    return text.strip()


def geeta_gpt(query):
    verse_pattern = re.search(r"chapter\s*(\d+)[^\d]+verse\s*(\d+)", query.lower())
    if verse_pattern:
        chapter, verse = verse_pattern.groups()
        chapter, verse = int(chapter), int(verse)
        if (chapter, verse) in verse_dict:
            d = verse_dict[(chapter, verse)].metadata
            return f"""**Chapter {chapter}, Verse {verse}**

📜 Sanskrit:
{d['sanskrit']}

🌍 Translation:
{d['english']}

🕉️ Krishna’s Guidance:
My dear Arjuna, perform your duty with sincerity but remain unattached to results. 🙏"""

    docs_with_scores = vector_db.similarity_search_with_score(query, k=5)
    relevant_docs = [d for d, score in docs_with_scores if score >= 0.6]

    verses_context = "\n\n".join([
        f"[Chapter {d.metadata['chapter']}, Verse {d.metadata['verse']}]\nSanskrit: {d.metadata['sanskrit']}\nEnglish: {d.metadata['english']}"
        for d in relevant_docs
    ])

    system_prompt = "You are GeetaGPT, the eternal voice of Shree Krishna.\n"

    prompt = f"{system_prompt}\nContext:\n{verses_context}\n\nUser Question: {query}\n\nAnswer as Shree Krishna:"
    raw_out = generator(prompt, max_new_tokens=350, temperature=0.3, do_sample=True)[0]["generated_text"]
    answer = raw_out.split("Answer as Shree Krishna:")[-1].strip()
    return clean_response(answer)


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🕉️ GeetaGPT")
st.write("Ask any question from the Bhagavad Gita or provide Chapter & Verse (e.g., 'Chapter 2, Verse 47')")

query = st.text_input("Your Question")
if st.button("Ask"):
    if query.strip():
        response = geeta_gpt(query)
        st.markdown(response)

