import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    ReadTheDocsLoader,
)
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain_openai import OpenAIEmbeddings


load_dotenv()


def ingest_catalog(path: str) -> None:
    """Use FAISS vector db to ingest a catalog of documents intro vectorstore"""
    print("Ingesting...")

    # There are way more loaders from langchain community
    loader = ReadTheDocsLoader(path)
    raw_documents = loader.load()
    print(f"{len(raw_documents)=}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=0
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"{len(documents)=}")

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace(
            os.path.basename(path), "https:/"
        )
        doc.metadata.update({"source": new_url})

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    print("ingesting...")

    vectorstore = FAISS.from_documents(
        documents, embeddings
    )
    vectorstore.save_local(
        f"{os.path.basename(path)}-faiss-vdb"
    )

    print("finish")


if __name__ == "__main__":
    ingest_catalog("langchain-docs")
