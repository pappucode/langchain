from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv

load_dotenv()

## Step 1a - Indexing (Document Ingestion)
video_id = "Gfr50f6ZBvo" # only the ID, not full URL

# transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
# for transcript in transcripts:
#     print(f"Language: {transcript.language_code} - {transcript.language}")

try:
  #if you don't care which language, this returns the "best" one
  transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

  # flatten it to plain text
  transcript = " ".join(chunk["text"] for chunk in transcript_list)
 #print(transcript)
 #print(transcript_list)

except TranscriptsDisabled:
  print("No captions available for this video.")

## Step 1b - Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = splitter.create_documents([transcript])

# print(len(chunks))
# print(chunks[0])

## Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")

# Continued at the script rag_using_langchain2.py