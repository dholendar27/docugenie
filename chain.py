import os
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Const
PERSIST_DIRECTORY = "vector_db"

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)


template = """You are a helpful AI assistant that answers questions.
If you don't know the answer, just say "I don't know". Don't try to make up an answer.
You are not allowed to answer from the online.
Use the following pieces of context to answer the question:
Context: 
{context}

Previous conversation:
{history}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    input_variables=["context", "history", "question"],
    template=template
)



text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

def split_documents(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    return text_splitter.split_documents(docs)

def convert_into_embeddings(filepath):
    docs = split_documents(filepath)
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="my_vector_db",
        persist_directory=PERSIST_DIRECTORY
    )


def chat(question):
    retriever = Chroma(
        collection_name="my_vector_db",
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    ).as_retriever(search_kwargs={"k": 3})

    # 2. Create Conversation Buffer Memory
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="question",
        return_messages=True  # Store chat history as messages
    )

    # 3. Create the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        chain_type_kwargs={
            "verbose": True,
            "prompt": PROMPT,
            "memory": memory,
        }
    )

    # 4. Run the chain with the user's question
    result = chain({"query": question})
    print(result)
    return result["result"]

