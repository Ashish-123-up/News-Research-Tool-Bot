# 📰 News Research Tool Bot

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to ask questions about any **news article** by simply pasting its URL. Built using **LangChain**, **Streamlit**, and **FAISS**, this tool acts like your personal research assistant — extracting, chunking, and understanding online news so you don’t have to.

---

## 🚀 Features

- 🔗 URL-Based Input
Users can input the URL of any news article. The bot automatically scrapes and processes the content behind the scenes.

-❓ Intelligent Question Answering
Ask anything related to the article — summaries, causes, impacts, key facts, timelines, etc. Powered by contextual understanding, not just keyword search.

-🧠 RAG Pipeline (Retrieval-Augmented Generation)
Combines retrieval of relevant content chunks with OpenAI’s LLM for natural language generation. Provides precise, grounded answers based on the article content.

-🤖 OpenAI Large Language Model Integration
Utilizes OpenAI’s gpt-3.5-turbo or gpt-4 for deep comprehension and high-quality responses to user queries.

-🧱 Efficient Text Chunking & Embedding
Breaks article text into semantically meaningful chunks, then embeds them using LangChain's embedding wrappers for optimized retrieval.

-🔍 Vector Search with FAISS
Uses FAISS for fast and scalable similarity search over embedded text — ensuring only relevant chunks are passed to the LLM.

-🖥️ Streamlit-Based User Interface
Clean and interactive UI built with Streamlit makes it easy for users to paste URLs, ask questions, and view answers instantly.

-🔐 Environment Variable Support
Uses .env file to safely manage sensitive credentials like OpenAI API keys — making the project production-ready.

