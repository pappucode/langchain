# LLMS - Generative AI Experiments and Tools

This repository contains my working experiments, utilities, and tools related to Large Language Models (LLMs) and Generative AI using Python, LangChain, OpenAI, and related technologies.

## Project Structure

The `llms/` directory includes:

- `genai/` - Virtual environment (NOTE: ignored from future commits)
- `langchain_chains/` - Experimenting LangChain chains
- `langchain_models/` - Woring with base models, chat models and embedding models both open source and closed source
- `langchain_output_parsers/` - Experimenting LangChain OutputParsers
- `langchain_structured_output/` - Experimenting LangChain Structured Output
- `prompts/` - Prompt engineering templates and techniques
- `rag/` - Retrieval-Augmented Generation implementations 
- `vectorstore/` - Vector DB experiments using Chroma, FAISS, etc.

## 📦 Setup

1. Create a virtual environment:
python -m venv venv

2. Activate it:
- On Windows: `venv\Scripts\activate`
- On macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
pip install -r requirements.txt

## Features & Experiments

- Prompt engineering examples
- RAG pipeline using LangChain
- Vector store integrations
- OpenAI GPT-3.5/4 API usage
- Web scraping with LangChain loaders
- Chatbot scaffolding (optional)

## API Key Management

API keys are handled securely using `.env` files. Be sure to set:
OPENAI_API_KEY=sk-xxx...