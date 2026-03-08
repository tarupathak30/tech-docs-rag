![Python](https://img.shields.io/badge/python-3.10-blue)
![FastAPI](https://img.shields.io/badge/framework-FastAPI-green)
![FAISS](https://img.shields.io/badge/vector--db-FAISS-orange)
![LLM](https://img.shields.io/badge/LLM-Groq-red)



TechDocs RAG System

A Retrieval-Augmented Generation system for answering technical documentation questions using semantic search and large language models.

Instead of letting a model guess answers, this system retrieves relevant documentation chunks, reranks them, and generates answers grounded strictly in the retrieved sources.

Built with FastAPI, FAISS, LangChain, and inference via Groq.



<h2 align="center">Architecture</h2>

<p align="center">
  <img src="images/architecture-diagram.png" width="750">
</p>
