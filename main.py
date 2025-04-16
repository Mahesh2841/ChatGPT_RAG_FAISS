from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader= WebBaseLoader('https://en.wikipedia.org/wiki/Carnegie_Mellon_University')
doc=loader.load()
splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs=splitter.split_documents(doc)

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
db=FAISS.from_documents(docs, embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'))

retriever=db.as_retriever(search_type='similarity', search_kwargs={'k':2})
