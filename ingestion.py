from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()


if __name__ == "__main__":
    print("Ingesting...")

    # There are way more loaders from langchain community
    loader = TextLoader("mediumblog.txt")
    document = loader.load()

    print("Splitting")

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    )
    texts = text_splitter.split_documents(document)
    print(f"creted {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    print("ingesting...")

    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ["INDEX_NAME"],
    )

    print("finish")
