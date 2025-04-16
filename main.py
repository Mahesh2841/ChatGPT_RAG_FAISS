from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Document Loader
loader= WebBaseLoader('https://en.wikipedia.org/wiki/Carnegie_Mellon_University')
doc=loader.load()
splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs=splitter.split_documents(doc)

#Create FAISS vectorstore
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
db=FAISS.from_documents(docs, embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'))


#Create a retriever
retriever=db.as_retriever(search_type='similarity', search_kwargs={'k':2})


#Create a Generator
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

os.environ['OPENAI_API_KEY']='your_openai_key'
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
template= """"
You are an assistant for question-answering tasks.
Use the provided context only to answer the following question:

<context>
{context}
</context>

Question: {question}
"""

#Set up the chain
prompt=PromptTemplate(template=template, input_variables=['context', 'question'])
chain=RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, chain_type_kwargs={"prompt": prompt})

#Inference
query="what is the full form of cmu?"
result = chain.invoke({"query": query})
print(result["result"])
