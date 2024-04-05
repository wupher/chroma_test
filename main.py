import os

from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

api_key = os.getenv('AIPROXY_API_KEY')
url_base = os.getenv('OPENAI_API_BASE')
mk_directory = './docs/api_documents'
index_directory = './docs/chroma'
embeddings = OpenAIEmbeddings(openai_api_key=api_key, openai_api_base=url_base)


def index_files():
    markdown_files = [os.path.join(mk_directory, filename) for filename in os.listdir(mk_directory) if
                      filename.endswith('.md')]
    documents = []
    for md_file in markdown_files:
        loader = UnstructuredMarkdownLoader(md_file)
        doc = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        documents.extend(text_splitter.split_documents(doc))
        print(f"load file: {md_file}")
    db = Chroma.from_documents(documents, embedding=embeddings, persist_directory=index_directory)
    db.persist()
    return db


def query_db(query, vector_db):
    return vector_db.similarity_search(query)


def loder_from_disk():
    db = Chroma(persist_directory=index_directory, embedding_function=embeddings)
    return db


if __name__ == '__main__':
    # chroma_db1 = index_files()
    question = "营业实时上报协议中的 gateway 是什么涵义？"
    # query_result = query_db(question, chroma_db1)
    # print(f"db1 answer: {query_result}")
    chroma_db2 = loder_from_disk()
    query_result2 = query_db(question, chroma_db2)
    print(f"db2 answer: {query_result2}")


