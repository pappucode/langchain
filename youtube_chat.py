import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.chat_history import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Helper Functions ---
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    return " ".join([chunk["text"] for chunk in transcript])

def build_chain(video_text):
    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([video_text])
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # History-aware retriever
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the chat history and question to retrieve relevant context."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(ChatOpenAI(), retriever, context_prompt)

    # Chain to answer from documents
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant answering questions from YouTube transcripts."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("system", "Context:\n{context}")
    ])
    document_chain = create_stuff_documents_chain(ChatOpenAI(), answer_prompt)

    final_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    # Wrap with memory
    store = {}
    memory_wrapper = RunnableWithMessageHistory(
        final_chain,
        lambda session_id: store.setdefault(session_id, ChatMessageHistory()),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return memory_wrapper

# --- Streamlit UI ---
st.title("🎥 YouTube Video Chat (LangChain Modern Version)")

video_url = st.text_input("Paste YouTube Video Link:")
session_id = "user-session"  # Simple static session, can be user ID or UUID for multi-user

if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        st.success(f"Fetched video ID: {video_id}")
        with st.spinner("Fetching transcript and preparing..."):
            text = get_transcript(video_id)
            chat_chain = build_chain(text)
        st.session_state.chat_chain = chat_chain
        st.session_state.ready = True
        st.success("Ready to chat!")

if st.session_state.get("ready", False):
    user_query = st.chat_input("Ask something about the video...")
    if user_query:
        with st.spinner("Thinking..."):
            response = st.session_state.chat_chain.invoke({"input": user_query}, config={"configurable": {"session_id": session_id}})
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(response["answer"])
