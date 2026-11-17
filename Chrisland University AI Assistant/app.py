import streamlit as st
import PyPDF2
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

st.set_page_config(page_title="Chrisland Assistant", page_icon="🎓", layout="wide")

DEEP_PURPLE = "#5A3178"
GOLD = "#E6A600"

st.markdown(f"""
<style>
    .main {{background-color: #FFFFFF;}}
    .stButton>button {{background-color: {DEEP_PURPLE}; color: white; font-weight: bold; border-radius: 8px; padding: 10px 25px; border: none;}}
    .stButton>button:hover {{background-color: {GOLD}; color: {DEEP_PURPLE};}}
    .header {{background: {DEEP_PURPLE}; padding: 30px; border-radius: 10px; text-align: center; color: white; margin-bottom: 25px;}}
    .stSidebar {{background-color: #F8F9FA;}}
    [data-testid="stSidebar"] {{background-color: #F8F9FA;}}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def read_pdf(path):
    text = ""
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + " "
    return re.sub(r'\s+', ' ', text.strip())

def create_ultra_chunks(text):
    """Creates 10,000+ chunks using multiple strategies"""
    chunks = []
    words = text.split()
    
    # Strategy 1: Small sliding windows (100 words, step 25)
    for i in range(0, len(words) - 100, 25):
        chunks.append(" ".join(words[i:i+100]))
    
    # Strategy 2: Medium sliding windows (200 words, step 50)
    for i in range(0, len(words) - 200, 50):
        chunks.append(" ".join(words[i:i+200]))
    
    # Strategy 3: Large sliding windows (300 words, step 75)
    for i in range(0, len(words) - 300, 75):
        chunks.append(" ".join(words[i:i+300]))
    
    # Strategy 4: Sentence-based chunks
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for i in range(0, len(sentences) - 5, 2):
        chunks.append(" ".join(sentences[i:i+5]))
    
    return [c for c in chunks if len(c) > 100]

@st.cache_resource
def build_index(_model, chunks):
    embeddings = []
    for i in range(0, len(chunks), 64):
        batch = _model.encode(chunks[i:i+64], convert_to_numpy=True)
        embeddings.append(batch)
    embeddings = np.vstack(embeddings)
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index

def search(query, model, index, chunks):
    qvec = model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(qvec)
    scores, indices = index.search(qvec, 50)
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if score > 0.30 and idx < len(chunks):
            results.append({'chunk': chunks[idx], 'score': float(score)})
    return sorted(results, key=lambda x: x['score'], reverse=True)

def make_answer(question, results):
    if not results:
        return "I couldn't find relevant information in the handbook. Please try rephrasing your question with different keywords."
    
    if results[0]['score'] < 0.35:
        return "The information found may not directly answer your question. Please be more specific or use exact terms from the handbook."
    
    # Extract question type and keywords
    q_lower = question.lower()
    is_what = any(w in q_lower for w in ['what', 'define', 'explain'])
    is_how = any(w in q_lower for w in ['how', 'process', 'procedure'])
    is_list = any(w in q_lower for w in ['requirements', 'rules', 'policies', 'steps'])
    
    # Get top relevant chunks
    top_chunks = [r['chunk'] for r in results[:8]]
    all_text = " ".join(top_chunks)
    
    # Extract sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', all_text) if len(s.strip().split()) >= 5]
    
    # Score sentences by relevance
    q_keywords = set(re.findall(r'\b\w{4,}\b', q_lower)) - {'what', 'when', 'where', 'which', 'tell', 'explain', 'describe', 'about', 'does'}
    
    scored_sents = []
    for sent in sentences:
        s_words = set(re.findall(r'\b\w{4,}\b', sent.lower()))
        overlap = len(q_keywords & s_words)
        
        # Bonus for definition/explanation patterns
        if is_what and any(p in sent.lower() for p in ['is', 'are', 'refers to', 'means', 'defined as']):
            overlap += 2
        if is_how and any(p in sent.lower() for p in ['must', 'should', 'will', 'can', 'process', 'step']):
            overlap += 2
        if is_list and any(c in sent for c in [':', '1.', '2.', '-']):
            overlap += 1
            
        if overlap > 0:
            scored_sents.append({'text': sent, 'score': overlap})
    
    scored_sents.sort(key=lambda x: x['score'], reverse=True)
    
    if not scored_sents:
        return all_text[:600] + "..."
    
    # Build structured answer
    selected = [s['text'] for s in scored_sents[:10]]
    
    # Format based on content type
    answer_parts = []
    current_para = []
    
    for sent in selected:
        # Check if sentence starts a list or new section
        if any(sent.strip().startswith(p) for p in ['1.', '2.', '3.', '•', '-']) or ':' in sent[:50]:
            if current_para:
                answer_parts.append(' '.join(current_para))
                current_para = []
            answer_parts.append(sent)
        else:
            current_para.append(sent)
            if len(' '.join(current_para)) > 250:
                answer_parts.append(' '.join(current_para))
                current_para = []
    
    if current_para:
        answer_parts.append(' '.join(current_para))
    
    # Join with proper spacing
    formatted = '\n\n'.join(answer_parts[:4])
    
    return formatted if formatted else ' '.join(selected[:6])

def main():
    st.markdown(f'<div class="header"><h1>Chrisland University Assistant</h1><p style="color: {GOLD};">Student Handbook AI Helper</p></div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown(f'<div style="text-align: center; padding: 25px 20px 20px 20px; border-bottom: 2px solid {DEEP_PURPLE};"><h1 style="color: {DEEP_PURPLE}; margin: 0; font-size: 1.8rem;">Chrisland AI</h1><p style="color: {DEEP_PURPLE}; margin: 8px 0 0 0; font-size: 0.8rem; opacity: 0.7;">Student Handbook Assistant</p></div>', unsafe_allow_html=True)
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        if 'ready' in st.session_state:
            c1, c2 = st.columns(2)
            c1.markdown(f'<div style="text-align: center;"><p style="color: {DEEP_PURPLE}; font-size: 0.7rem; margin: 0; opacity: 0.6;">STATUS</p><p style="color: {DEEP_PURPLE}; font-size: 1.2rem; margin: 3px 0; font-weight: 600;">Online</p></div>', unsafe_allow_html=True)
            c2.markdown(f'<div style="text-align: center;"><p style="color: {DEEP_PURPLE}; font-size: 0.7rem; margin: 0; opacity: 0.6;">SECTIONS</p><p style="color: {DEEP_PURPLE}; font-size: 1.2rem; margin: 3px 0; font-weight: 600;">{len(st.session_state.chunks):,}</p></div>', unsafe_allow_html=True)
        
        st.markdown(f"<div style='height: 25px;'></div><div style='padding: 0 10px;'><h3 style='color: {DEEP_PURPLE}; font-size: 0.95rem; margin: 0 0 10px 0;'>Quick Tips</h3><p style='color: {DEEP_PURPLE}; font-size: 0.8rem; margin: 6px 0; opacity: 0.8;'>Be specific with keywords</p><p style='color: {DEEP_PURPLE}; font-size: 0.8rem; margin: 6px 0; opacity: 0.8;'>Ask one topic at a time</p></div><div style='height: 22px;'></div><div style='padding: 0 10px;'><h3 style='color: {DEEP_PURPLE}; font-size: 0.95rem; margin: 0 0 10px 0;'>Try Asking</h3><p style='color: {DEEP_PURPLE}; font-size: 0.8rem; margin: 6px 0; opacity: 0.8;'>What are admission requirements?</p><p style='color: {DEEP_PURPLE}; font-size: 0.8rem; margin: 6px 0; opacity: 0.8;'>Explain the grading system</p></div>", unsafe_allow_html=True)
    
    if not os.path.exists("student_handbook.pdf"):
        st.error("student_handbook.pdf not found!")
        return
    
    if 'ready' not in st.session_state:
        with st.spinner("Building ultra-massive knowledge base... 1-2 minutes..."):
            text = read_pdf("student_handbook.pdf")
            chunks = create_ultra_chunks(text)
            model = load_model()
            index = build_index(model, chunks)
            
            st.session_state.update({'model': model, 'index': index, 'chunks': chunks, 'ready': True, 'messages': []})
            st.success(f"Ready! {len(chunks):,} sections loaded")
            st.balloons()
    
    st.markdown("---\n## Ask Your Question")
    
    for msg in st.session_state.messages:
        st.markdown(f"**You:** {msg['content']}" if msg['role'] == 'user' else f"**Assistant:**")
        if msg['role'] == 'assistant':
            st.info(msg['content'])
        st.markdown("")
    
    c1, c2 = st.columns([5, 1])
    question = c1.text_input("Type your question:", key="q", placeholder="e.g., What are the admission requirements?", label_visibility="collapsed")
    ask = c2.button("Ask", use_container_width=True)
    
    if ask and question:
        st.session_state.messages.append({'role': 'user', 'content': question})
        with st.spinner("Searching..."):
            results = search(question, st.session_state.model, st.session_state.index, st.session_state.chunks)
            answer = make_answer(question, results)
        st.session_state.messages.append({'role': 'assistant', 'content': answer})
        st.rerun()
    
    c1, c2 = st.columns(2)
    if c1.button("Clear Chat", use_container_width=True) if st.session_state.messages else False:
        st.session_state.messages = []
        st.rerun()
    if c2.button("Reload", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()