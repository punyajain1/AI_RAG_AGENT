# AI Agent : Context-Aware Question Answering from document 🤖📄

This project is a command-line AI assistant that answers user questions based on a collection of PDF and text document. It uses OpenAI's GPT-4o model and vector search (ChromaDB) to retrieve relevant context from your document and provide concise, context-aware answers.

## Features ✨
- **PDF/Text Ingestion:** Automatically loads and processes all PDF and text files in the `files/` directory.
- **Vector Store:** Embeds and stores document chunks using OpenAI embeddings and ChromaDB for fast retrieval.
- **Contextual Q&A:** Answers questions using only the information found in your document. If a question is out of context, the assistant will say so.
- **Chat History Awareness:** Reformulates follow-up questions to be self-contained using chat history.
- **Command-Line Chat:** Interactive chat loop in the terminal.

## Setup 🛠️

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   (You may need to create `requirements.txt` with the following packages: `langchain`, `openai`, `chromadb`, `PyPDF2`, `python-dotenv`)

3. **Set up your OpenAI API key:**
   - Create a `.env` file in the project root:
     ```env
     OPENAI_API_KEY=your_openai_api_key_here
     ```

4. **Add your document:**
   - Place your PDF or text files in the `files/` directory.

5. **Run the assistant:**
   ```bash
   python app.py
   ```

## Usage 💬
- Start chatting with the AI in your terminal.
- Type your question and press Enter.
- Type `exit` to end the conversation.

## Project Structure 🗂️
```
app.py                # Main application file
files/                # Place your PDF and text document here
db/chroma_db_document/   # Vector store database (auto-generated)
.env                  # Your OpenAI API key (not included in repo)
```

## document
- The assistant will only answer questions that can be answered from your document. If a question is out of context, it will say so.
- Supports follow-up questions by reformulating them using chat history.
