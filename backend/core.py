import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.chains.retrieval import (
    create_retrieval_chain,
)
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain.chains.history_aware_retriever import (
    create_history_aware_retriever,
)
from langchain import hub
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS


load_dotenv()


def call_llm_with_rag(
    query: str,
    vector_db_path: str,
    chat_history: List[Dict[str, Any]] = [],
) -> Dict[str, Any]:
    """Calls LLM with RAG"""

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    docsearch = FAISS.load_local(
        vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retrieval_qa_chat_prompt = hub.pull(
        "langchain-ai/retrieval-qa-chat"
    )

    stuff_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )

    rephrapse_prompt = hub.pull(
        "langchain-ai/chat-langchain-rephrase"
    )

    history_retriever = create_history_aware_retriever(
        OpenAI(),
        retriever=docsearch.as_retriever(),
        prompt=rephrapse_prompt,
    )

    qa = create_retrieval_chain(
        history_retriever,
        combine_docs_chain=stuff_docs_chain,
    )
    result = qa.invoke(
        input={"input": query, "chat_history": chat_history}
    )

    out = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return out


if __name__ == "__main__":
    res = call_llm_with_rag(
        "What is a chain?", "langchain-docs-faiss-vdb/"
    )
    print(type(res))
    print(res["answer"])
