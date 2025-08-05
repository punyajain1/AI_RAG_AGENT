import PyPDF2
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv


load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
presistence_dir = os.path.join(current_dir,"db","chroma_db_document")
notes_dir = os.path.join(current_dir, "files")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(presistence_dir):
    os.makedirs(presistence_dir)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")


print(f"Notes directory exists: {os.path.exists(notes_dir)}")
print(f"DB directory exists: {os.path.exists(db_dir)}")
if os.path.exists(notes_dir):
    print(f"Files in notes directory: {os.listdir(notes_dir)}")

if os.path.exists(notes_dir):
    notes_documents = []
    for filename in os.listdir(notes_dir):
        file_path = os.path.join(notes_dir, filename)
        if os.path.isfile(file_path):
            try:
                if filename.lower().endswith('.pdf'):
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page_num in range(len(pdf_reader.pages)):
                            text += pdf_reader.pages[page_num].extract_text() + "\n"
                        notes_documents.append(Document(page_content=text, metadata={"source": filename}))
                else:
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        notes_documents.append(Document(page_content=content, metadata={"source": filename}))
                print(f"Added {filename} to the vector database")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    if notes_documents:
        rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        rec_char_docs = rec_char_splitter.split_documents(notes_documents)
        db = Chroma.from_documents(rec_char_docs, embeddings, persist_directory=presistence_dir)
        db.persist()  
    else:
        print("No files found in notes folder")
else:
    print(f"Notes directory not found at {notes_dir}")





def initialize_or_load_vector_store():
    if os.path.exists(presistence_dir):
        print("Loading existing vector store...")
        db = Chroma(persist_directory=presistence_dir, embedding_function=embeddings)
    else:
        print("Creating new vector store...")
        os.makedirs(presistence_dir, exist_ok=True)
        db = Chroma(persist_directory=presistence_dir, embedding_function=embeddings)
    return db

db = initialize_or_load_vector_store()

retriever = db.as_retriever( search_type="mmr", search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5})


contextualize_q_system_prompt = (
    "Given a chat history and the latest user message, which may refer to earlier parts of the conversation:"
    "Rewrite the message as a standalone question that can be fully understood without needing the chat history."
    "Do not answer the question—your task is only to reformulate it."
    "If the message is already clear and self-contained, return it as is."
    "If the message is unclear or ambiguous, ask the user for clarification."
    "If the message is out of context and cannot be understood without additional background, respond with:"
        "The question is out of context and cannot be reformulated."
    "The reformulated question should be clear, concise, and specific."
)


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

#history aware chain
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_system_prompt = (
    "You are an assistant for question-answering from data given. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use five sentences maximum and keep the answer "
    "be concise."
    "dont answer if it is out of the context, say 'The question is out of context and cannot be answered.'"
    "\n\n"
    "{context}"
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break
        
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        
        print(f"\nAI: {result['answer']}\n")
        
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"])) 


if __name__ == "__main__":
    continual_chat()