# geeta_streamlit.py
import streamlit as st
import pandas as pd
import re
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import pipeline

# ---------------------------
# 1. Load FAISS Index & Verse Dict
# ---------------------------
@st.cache_data
def load_vector_db_and_verses(csv_path="geeta_dataset.csv", index_path="geeta_faiss_index"):
    try:
        vector_db = FAISS.load_local(
            index_path,
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        df = pd.read_csv(csv_path)
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
    except:
        # Build index if not exists
        df = pd.read_csv(csv_path)
        docs = []
        verse_dict = {}
        for _, row in df.iterrows():
            doc = Document(
                page_content=row["english"],
                metadata={
                    "chapter": int(row.chapter),
                    "verse": int(row.verse),
                    "sanskrit": row.sanskrit.strip(),
                    "english": row.english.strip(),
                    "id": f"{row.chapter}.{row.verse}"
                }
            )
            docs.append(doc)
            verse_dict[(int(row.chapter), int(row.verse))] = doc
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(docs, embeddings)
        vector_db.save_local(index_path)
    return vector_db, verse_dict

vector_db, verse_dict = load_vector_db_and_verses()

# ---------------------------
# 2. Load LLaMA 3 1B Generator
# ---------------------------
@st.cache_resource
def load_generator():
    generator = pipeline(
        "text-generation",
        model="meta-llama/Llama-3-1B-Instruct",  # 1B model works on free Spaces/CPU
        device=-1  # CPU-friendly
    )
    return generator

generator = load_generator()

# ---------------------------
# 3. Clean response utility
# ---------------------------
def clean_response(text: str) -> str:
    text = re.sub(r"(🙏\s*){2,}", "🙏", text)
    text = re.sub(r"(May this wisdom guide you[^\n]*){2,}", "May this wisdom guide you. 🙏", text)
    return text.strip()

# ---------------------------
# 4. GeetaGPT function
# ---------------------------
def geeta_gpt(query, vector_db, verse_dict, top_k=4, similarity_threshold=0.7):
    verse_pattern = re.search(r"chapter\s*(\d+)[^\d]+verse\s*(\d+)", query.lower())
    if verse_pattern:
        chapter, verse = verse_pattern.groups()
        chapter = int(chapter)
        verse = int(verse)
        if (chapter, verse) in verse_dict:
            d = verse_dict[(chapter, verse)].metadata
            return f"""**Chapter {chapter}, Verse {verse}**

📜 *Sanskrit*:
{d.get('sanskrit', 'Sanskrit text unavailable')}

🌍 *Translation*:
{d.get('english', 'English translation unavailable')}

🕉️ *Krishna’s Guidance*:
My dear Arjuna, reflect on this teaching. Perform your duty with sincerity, but remain unattached to the fruits. May this wisdom guide you. 🙏"""

    docs_with_scores = vector_db.similarity_search_with_score(query, k=top_k)
    relevant_docs = [d for d, score in docs_with_scores if score >= similarity_threshold]

    verses_context = "\n\n".join([
        f"[Chapter {d.metadata['chapter']}, Verse {d.metadata['verse']}]\n"
        f"Sanskrit: {d.metadata['sanskrit']}\nEnglish: {d.metadata['english']}"
        for d in relevant_docs
    ])

    system_prompt = (
        "You are GeetaGPT, the eternal voice of Shree Krishna from the Bhagavad Gita. "
        "Answer questions with clarity, compassion, and authority. "
        "Always cite relevant verses if available. "
        "End with 'May this wisdom guide you. 🙏'\n"
    )

    prompt = f"""{system_prompt}

Context (Bhagavad Gita verses):
{verses_context}

User's Question: {query}

Answer as Shree Krishna:
"""
    raw_out = generator(prompt, max_new_tokens=250, temperature=0.3, do_sample=True)[0]["generated_text"]
    answer = raw_out.split("Answer as Shree Krishna:")[-1].strip()
    return clean_response(answer)

# ---------------------------
# 5. Streamlit UI
# ---------------------------
st.title("🕉️ GeetaGPT")
st.markdown("Ask any question about life or the Bhagavad Gita, and get guidance from Krishna.")

user_input = st.text_area("Your Question:", height=120)
if st.button("Ask Krishna"):
    if user_input.strip():
        with st.spinner("Krishna is answering..."):
            response = geeta_gpt(user_input, vector_db, verse_dict)
        st.markdown(response)
