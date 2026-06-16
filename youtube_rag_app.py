"""
YouTube RAG Chatbot
Tested with: langchain==1.3.8, langchain-community==0.4.2
- langchain.retrievers does NOT exist
- MergerRetriever, MultiQueryRetriever, ContextualCompressionRetriever
  are all implemented manually or sourced from langchain_core
"""

import os, re, warnings
warnings.filterwarnings("ignore")

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List
from pydantic import Field

# ── RetrievalQA replacement using LCEL (no langchain.chains needed) ───────────
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Manual MergerRetriever (deduplicates across multiple retrievers) ──────────
class MergerRetriever(BaseRetriever):
    retrievers: List[BaseRetriever] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        seen, docs = set(), []
        for retriever in self.retrievers:
            try:
                for doc in retriever.invoke(query):
                    key = doc.page_content[:100]
                    if key not in seen:
                        seen.add(key)
                        docs.append(doc)
            except Exception:
                pass
        return docs

# ── Manual MultiQueryRetriever (generates multiple queries via LLM) ───────────
class MultiQueryRetriever(BaseRetriever):
    retriever: BaseRetriever
    llm: BaseLanguageModel

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        prompt = f"""Generate 3 different versions of this question to retrieve relevant documents.
Return only the questions, one per line, no numbering.
Original question: {query}"""
        try:
            response = self.llm.invoke(prompt)
            queries = [query] + [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        except Exception:
            queries = [query]

        seen, docs = set(), []
        for q in queries:
            try:
                for doc in self.retriever.invoke(q):
                    key = doc.page_content[:100]
                    if key not in seen:
                        seen.add(key)
                        docs.append(doc)
            except Exception:
                pass
        return docs

# ── Manual ContextualCompressionRetriever ────────────────────────────────────
class ContextualCompressionRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    llm: BaseLanguageModel

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        compressed = []
        for doc in docs:
            try:
                prompt = f"""Extract only the parts of the following text relevant to the question.
If nothing is relevant, reply with "IRRELEVANT".

Question: {query}
Text: {doc.page_content}

Relevant extract:"""
                response = self.llm.invoke(prompt)
                content = response.content.strip()
                if content and content.upper() != "IRRELEVANT":
                    compressed.append(Document(page_content=content, metadata=doc.metadata))
            except Exception:
                compressed.append(doc)
        return compressed if compressed else docs

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    .main { background: #0f0f0f; }
    .block-container { padding: 2rem 2.5rem; max-width: 960px; }
    h1 { color: #ff4444 !important; font-family: 'Segoe UI', sans-serif; }
    .stTextInput > div > div > input { background: #1e1e1e; color: #f0f0f0; border: 1px solid #333; border-radius: 8px; }
    .user-msg { background: #1a3a5c; color: #e8f4fd; padding: 12px 16px; border-radius: 16px 16px 4px 16px; margin: 8px 0 8px 15%; font-size: 0.95rem; }
    .bot-msg  { background: #1e1e1e; color: #e0e0e0; padding: 12px 16px; border-radius: 16px 16px 16px 4px; margin: 8px 15% 8px 0; border-left: 3px solid #ff4444; font-size: 0.95rem; }
    .source-box { background: #111; border: 1px solid #333; border-radius: 8px; padding: 8px 12px; font-size: 0.78rem; color: #888; margin-top: 4px; }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-right: 6px; }
    .badge-green  { background: #0d3321; color: #4ade80; }
    .badge-blue   { background: #0d1f3c; color: #60a5fa; }
    .badge-yellow { background: #2a1f0a; color: #fbbf24; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in {
    "chat_history": [], "vector_store": None,
    "bm25_retriever": None, "video_title": "",
    "ragas_scores": [], "current_url": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_video_id(url: str):
    for p in [r"(?:v=|\/)([0-9A-Za-z_-]{11})", r"youtu\.be\/([0-9A-Za-z_-]{11})"]:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

DOMAIN_KEYWORDS = {
    "technical": ["code", "programming", "algorithm", "software", "function", "api", "data"],
    "science":   ["physics", "chemistry", "biology", "experiment", "research"],
    "history":   ["history", "war", "ancient", "century", "civilization"],
}
def detect_domain(query: str) -> str:
    q = query.lower()
    for domain, kws in DOMAIN_KEYWORDS.items():
        if any(k in q for k in kws):
            return domain
    return "general"

def get_domain_instruction(domain: str) -> str:
    return {
        "technical": "Focus on technical accuracy. Use precise terminology.",
        "science":   "Explain scientific concepts clearly with evidence from context.",
        "history":   "Provide historical context and chronological accuracy.",
        "general":   "Answer conversationally and helpfully.",
    }.get(domain, "Answer helpfully.")

def get_embeddings(gemini_key: str, task_type: str = "retrieval_document"):
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=gemini_key,
        task_type=task_type,
    )

def get_video_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()
        for lang in ["en", "hi", "te"]:
            try:
                transcript = api.fetch(video_id, languages=[lang])
                return " ".join(snippet.text for snippet in transcript)
            except Exception:
                continue
        transcript = api.fetch(video_id)
        return " ".join(snippet.text for snippet in transcript)
    except Exception as e:
        raise RuntimeError(f"Unable to retrieve transcript.\n\n{str(e)}")

# ── Build index ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_index(youtube_url: str, gemini_key: str):
    os.environ["GOOGLE_API_KEY"] = gemini_key
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Could not extract video ID from URL.")
    full_text = get_video_transcript(video_id)
    if not full_text.strip():
        raise ValueError("Transcript is empty.")
    title = f"YouTube Video ({video_id})"
    docs = [Document(page_content=full_text, metadata={"source": youtube_url, "title": title})]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " "])
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, get_embeddings(gemini_key))
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 4
    return vector_store, bm25, title, len(chunks)

# ── Build retriever pipeline ──────────────────────────────────────────────────
def build_retriever(vector_store, bm25_retriever, llm):
    # 1. Dense MMR retriever
    dense = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 6, "fetch_k": 12, "lambda_mult": 0.6}
    )
    # 2. Multi-query on dense
    multi = MultiQueryRetriever(retriever=dense, llm=llm)
    # 3. Hybrid merge with BM25
    merged = MergerRetriever(retrievers=[multi, bm25_retriever])
    # 4. Contextual compression
    compressed = ContextualCompressionRetriever(base_retriever=merged, llm=llm)
    return compressed

# ── Answer generation using LCEL ─────────────────────────────────────────────
def get_answer(vector_store, bm25_retriever, domain: str, query: str, gemini_key: str):
    os.environ["GOOGLE_API_KEY"] = gemini_key
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=gemini_key)

    retriever = build_retriever(vector_store, bm25_retriever, llm)
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""You are an expert assistant for YouTube video content.
{get_domain_instruction(domain)}

Use ONLY the following context extracted from the video transcript to answer.
If the answer is not in the context, say "I couldn't find that in the video."

Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    return response.content.strip(), docs

# ── Ragas ─────────────────────────────────────────────────────────────────────
def evaluate_with_ragas(question, answer, contexts, gemini_key):
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
        os.environ["GOOGLE_API_KEY"] = gemini_key
        result = evaluate(
            Dataset.from_dict({"question": [question], "answer": [answer],
                               "contexts": [contexts], "ground_truth": [answer]}),
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=gemini_key),
            embeddings=get_embeddings(gemini_key),
        )
        return {"faithfulness": round(result["faithfulness"], 3),
                "answer_relevancy": round(result["answer_relevancy"], 3),
                "context_precision": round(result["context_precision"], 3)}
    except Exception as e:
        return {"error": str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🎬 YouTube RAG Chatbot")
st.markdown("*Ask anything about a YouTube video using advanced RAG*")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    gemini_key    = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
    langsmith_key = st.text_input("LangSmith API Key (optional)", type="password", placeholder="ls__...")
    enable_ragas     = st.toggle("Enable Ragas Evaluation", value=False)
    enable_langsmith = st.toggle("Enable LangSmith Tracing", value=False)
    if enable_langsmith and langsmith_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"]    = langsmith_key
        os.environ["LANGCHAIN_PROJECT"]    = "youtube-rag-chatbot"
    st.divider()
    st.markdown("### 🔧 Pipeline Features")
    st.markdown("""
<span class='badge badge-blue'>Multi-Query</span> Pre-retrieval<br><br>
<span class='badge badge-green'>MMR + BM25</span> Hybrid retrieval<br><br>
<span class='badge badge-yellow'>Merger</span> Dedup retrieval<br><br>
<span class='badge badge-blue'>Compression</span> Post-retrieval<br>
""", unsafe_allow_html=True)
    if st.session_state.video_title:
        st.divider()
        st.markdown(f"**📺 Loaded:**  \n{st.session_state.video_title}")
    if st.session_state.ragas_scores:
        st.divider()
        st.markdown("### 📊 Last Ragas Score")
        s = st.session_state.ragas_scores[-1]
        if "error" not in s:
            st.metric("Faithfulness",      s.get("faithfulness", "–"))
            st.metric("Answer Relevancy",  s.get("answer_relevancy", "–"))
            st.metric("Context Precision", s.get("context_precision", "–"))
        else:
            st.warning(f"Ragas error: {s['error']}")

col1, col2 = st.columns([4, 1])
with col1:
    youtube_url = st.text_input("🔗 YouTube URL", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")
with col2:
    load_btn = st.button("Load Video", use_container_width=True, type="primary")

if load_btn:
    if not gemini_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not youtube_url or not extract_video_id(youtube_url):
        st.error("Please enter a valid YouTube URL.")
    elif youtube_url == st.session_state.current_url:
        st.info("This video is already loaded. Start chatting below!")
    else:
        with st.spinner("📥 Loading transcript and building index…"):
            try:
                vs, bm25, title, n_chunks = build_index(youtube_url, gemini_key)
                st.session_state.update({
                    "vector_store": vs, "bm25_retriever": bm25,
                    "video_title": title, "current_url": youtube_url,
                    "chat_history": [],
                })
                st.success(f"✅ **{title}** indexed — {n_chunks} chunks ready.")
            except Exception as e:
                st.error(f"Failed to load video: {e}")

st.divider()

with st.container():
    for turn in st.session_state.chat_history:
        st.markdown(f"<div class='user-msg'>🧑 {turn['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-msg'>🤖 {turn['bot']}</div>",  unsafe_allow_html=True)
        if turn.get("sources"):
            with st.expander("📄 Source chunks", expanded=False):
                for i, src in enumerate(turn["sources"], 1):
                    st.markdown(f"<div class='source-box'><b>Chunk {i}</b><br>{src[:300]}…</div>", unsafe_allow_html=True)

if st.session_state.vector_store:
    with st.form("chat_form", clear_on_submit=True):
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            user_query = st.text_input("Ask a question…", label_visibility="collapsed",
                                       placeholder="e.g. What is the main topic discussed?")
        with btn_col:
            send = st.form_submit_button("Send ➤", use_container_width=True)
    if send and user_query.strip():
        if not gemini_key:
            st.error("Gemini API key required.")
        else:
            with st.spinner("🔍 Retrieving and generating answer…"):
                try:
                    answer, source_docs = get_answer(
                        st.session_state.vector_store,
                        st.session_state.bm25_retriever,
                        detect_domain(user_query),
                        user_query,
                        gemini_key,
                    )
                    sources = [d.page_content for d in source_docs]
                    st.session_state.chat_history.append({"user": user_query, "bot": answer, "sources": sources})
                    if enable_ragas and sources:
                        st.session_state.ragas_scores.append(
                            evaluate_with_ragas(user_query, answer, sources, gemini_key))
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
else:
    st.info("⬆️  Load a YouTube video above to start chatting.")

st.divider()
st.markdown("<center style='color:#555; font-size:0.8rem;'>YouTube RAG Chatbot · Multi-Query · MMR · BM25 · Merger · Contextual Compression · Ragas</center>", unsafe_allow_html=True)
