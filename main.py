import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY is missing from .env"
    )


loader = TextLoader(
    "data/ai_notes.txt",
    encoding = "utf-8"
)

documents = loader.load()
print("Documents loaded.")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
chunks = splitter.split_documents(
    documents
)
print(
    f"Created {len(chunks)} chunks."
)

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded.")

#CREATE FAISS AND LOAD FAISS

faiss_path = "faiss_db"
if os.path.exists(faiss_path):
    print("Loading existing FAISS database.")

    vectorstore = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Creating FAISS database")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(
        faiss_path
    )
    print("FAISS database saved.")


retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=groq_api_key
)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the question using ONLY the provided context.

If the answer is not present in the context,
say:

"I don't know based on the provided documents."

Do not invent information.

Context:
{context}

Question:
{question}

Answer:
"""
)

question = input(
    "\nAsk a question: "
)

retrieved_docs = retriever.invoke(
    question
)

context = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)

prompt_value = prompt.invoke(
    {
        "context":context,
        "question": question
    }
)

response = llm.invoke(
    prompt_value
)


print("\n========================================")
print("RETRIEVED DOCUMENTS")
print("========================================")

for i, doc in enumerate(
    retrieved_docs,
    start=1
):

    print(f"\n--- Document {i} ---")
    print(doc.page_content)


print("\n========================================")
print("ANSWER")
print("========================================")

print(response.content)
