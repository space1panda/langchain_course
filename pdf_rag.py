import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain.chains.retrieval import (
    create_retrieval_chain,
)
from langchain_community.vectorstores import FAISS


load_dotenv()


if __name__ == "__main__":
    print("Storing data...")

    print("Read pdf file...")

    pdf_path = "react.pdf"
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    print("loaded")

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30
    )
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    print("ingesting...")

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("faiss-react-store")

    print("Vector dataset created")

    """ Running RAG """

    print("Retrieving...")

    loaded_vdb = FAISS.load_local(
        "faiss-react-store",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retrieval_qa_chat_prompt = hub.pull(
        "langchain-ai/retrieval-qa-chat"
    )
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )

    query = "Explain to me in 4 sentences what is ReAct"
    result = retrieval_chain.invoke(input={"input": query})
    print(result["answer"])
