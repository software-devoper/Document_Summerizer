from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import PromptTemplate
import pandas as pd
import tempfile 
import os
from typing import Optional, Tuple, List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize the model
@st.cache_resource
def Load_model():
    try:
        return ChatGoogleGenerativeAI(
            model='gemini-2.5-flash', 
            temperature=0.7,
            max_retries=3
        )
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Generate Embeddings
@st.cache_resource
def Load_Embed():
    try:
        return HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        st.error(f"Failed to load embeddings: {str(e)}")
        return None

# Initialize the parser
parser = StrOutputParser()

# Initialize the session state
def init_session_state():
    default_states = {
        'Chat_history': [],
        'document_loaded': False,
        'document_text': "",
        'document_name': '',
        'document_type': '',
        'query': '',
        'info': '',
        'auto_query': '',
        'vector_store': None,
        'file_processed': False
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,  
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

# Get content based on the length of the context
def get_context(chunk: List, query: str = "") -> str:
    try:
        embedding = Load_Embed()
        if embedding is None:
            return "Error: Embeddings not available"
            
        total_text = "\n".join([page.page_content for page in chunk])
        st.session_state.info = total_text
        
        if len(total_text) < 150000 and not query: 
            return total_text
        else:
            if not query:
                query = "overview"
                
            if st.session_state.vector_store is None:
                st.session_state.vector_store = FAISS.from_documents(chunk, embedding)
            
            retriever = st.session_state.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 15, "lambda_mult": 0.5}
            )
            res = retriever.invoke(query)
            context = "\n".join([doc.page_content for doc in res])
            return context
    except Exception as e:
        logger.error(f"Error in get_context: {str(e)}")
        return f"Error processing context: {str(e)}"

def Load_document(file) -> Tuple[Optional[str], str]:
    try:
        if isinstance(file, str):  
            file_extension = os.path.splitext(file)[1].lower()
            temp_file_path = file
        else:
            file_extension = os.path.splitext(file.name)[1].lower()
            st.session_state.document_name = file.name
            st.session_state.document_type = file_extension.upper().replace('.', '')
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(file.getvalue())
                temp_file_path = temp_file.name

        splitter = get_text_splitter()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            if not docs:
                return None, "No content found in PDF"
            chunks = splitter.split_documents(docs)
            content = get_context(chunks)
            
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path, encoding='utf-8')
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            content = get_context(chunks)
            
        elif file_extension == '.csv':
            loader = CSVLoader(temp_file_path)
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            content = get_context(chunks)
            
            df = pd.read_csv(temp_file_path)
            st.session_state.dataframe = df
            
        else:
            return None, "Unsupported file format"

        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        st.session_state.file_processed = True
        return content, f"Successfully loaded {st.session_state.document_name}"
        
    except Exception as e:
        logger.error(f"Error loading document: {str(e)}")
        # Clean up temp file in case of error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return None, f"Error loading file: {str(e)}"

# Improved answer generation with better error handling
def get_answer(query: str) -> str:
    try:
        model = Load_model()
        if model is None:
            return "Error: Model not available"
            
        if not st.session_state.document_text:
            return "No document content available. Please upload a document first."

        template = PromptTemplate(
            template="""
You are an intelligent assistant that reads and understands documents.

Below is the extracted content from a document:

---------------------
{content}
---------------------

Based on the above content, answer the following user query clearly and accurately:

User Query: {query}

Important Instructions:
- Answer based ONLY on the provided document content
- If the answer cannot be found in the document, reply: "The answer is not available in the document."
- Be specific and provide relevant details from the document
- Format your response in a clear, readable way using bullet points or numbered lists when appropriate
- If the query is complex, break down your answer into logical sections

Answer:
            """,
            input_variables=['content', 'query']
        )

        chain = template | model | parser
        with st.spinner("Analyzing document and generating answer..."):
            result = chain.invoke({
                'content': st.session_state.document_text,
                'query': query
            })
            return result
            
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}"

def clear_chat_history():
    """Clear chat history while keeping document state"""
    st.session_state.Chat_history = []
    st.session_state.auto_query = ''
    st.session_state.query = ''

def main():
    st.set_page_config(
        page_title="Smart Document Q&A Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Enhanced CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
    }
        .chat-message {
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 12px;
            color: #fff;
            word-wrap: break-word;
        }

        .user-message {
            border-left: 4px solid #2196f3;
        }

        .assistant-message {
            border-left: 4px solid #9c27b0;
        }
    .document-info {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: bold;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📚 Smart Document Q&A Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Ask questions about your PDF, TXT, or CSV files")
    
    # Sidebar
    with st.sidebar:
        st.markdown("---")
        st.header("📁 Document Upload")
        
        upload_file = st.file_uploader(
            "Choose a Document",
            type=['pdf', 'txt', 'csv'],
            help='Upload PDF, TXT or CSV files to ask questions about their content'
        )
        
        if upload_file is not None:
            if (not st.session_state.document_loaded or 
                upload_file.name != st.session_state.document_name):
                
                with st.spinner("Loading document..."):
                    content, msg = Load_document(upload_file)
                    
                if content:
                    st.session_state.document_text = content
                    st.session_state.document_loaded = True
                    st.success(msg)
                    
                    # Clear previous chat when new document is loaded
                    clear_chat_history()
                    
                    st.markdown('---')
                    st.subheader("📊 Document Information")    
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Characters", len(st.session_state.info))
                    with col2:
                        st.metric("Words", len(st.session_state.info.split()))    
                    
                    st.info(f"**File Type:** {st.session_state.document_type}")
                    st.info(f"**File Name:** {st.session_state.document_name}")  
                    
                    # Show document preview
                    with st.expander("📄 Document Preview", expanded=False):
                        preview_text = (st.session_state.info[:800] + "..." 
                                      if len(st.session_state.info) > 800 
                                      else st.session_state.info)
                        st.text_area("Preview", preview_text, height=200, 
                                   label_visibility="collapsed", key="preview")
                else:
                    st.error(msg)
        else:
            st.info("👆 Please upload a document to get started!")
            
            # Example questions
            st.markdown("---")
            st.subheader("💡 Example Questions")
            st.markdown("""
            Once you upload a document, you can ask:
            - *What is the main topic of this document?*
            - *Summarize the key points*
            - *Find specific information about...*
            - *List all the important dates/names*
            - *What are the conclusions?*
            """)
        
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        This AI-powered assistant can read and understand your documents. 
        Supported formats:
        - **PDF** documents
        - **TXT** text files  
        - **CSV** data files
        
        Simply upload a file and start asking questions!
        """)
    # Main content area    
    if not st.session_state.document_loaded:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("## 🎯 How to Use")
            st.markdown("""
            1. **Upload a Document**: Use the sidebar to upload your PDF, TXT, or CSV file
            2. **Ask Questions**: Once loaded, ask any question about the document content
            3. **Get Answers**: The AI will analyze the document and provide accurate answers
            
            ### 📋 Supported Features:
            - **Document Q&A**: Ask specific questions about your documents
            - **Document Summary**: Get quick summaries of long documents
            - **Key Information Extraction**: Find important facts and data
            - **Multi-format Support**: Works with PDFs, text files, and CSV data
            """)
            
        with col2:
            st.markdown("## 🚀 Quick Start")
            st.markdown("""
            **Get started in seconds:**
            
            1. Click **'Browse files'** in sidebar
            2. Select your document
            3. Wait for upload confirmation
            4. Start asking questions!
            """)
            st.markdown("---")
            st.markdown("### 💬 Sample Questions")
            st.markdown("""
            - *"What is this document about?"*
            - *"Summarize the main points"*
            - *"Find information about [topic]"*
            - *"List the key findings"*
            - *"What are the recommendations?"*
            """)
    else:
        # Document is loaded - show Q&A interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="sub-header">💬 Ask Questions</div>', unsafe_allow_html=True)
            
            # Quick action buttons
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                if st.button("📋 Summarize", use_container_width=True):
                        st.session_state.auto_query = "Provide a comprehensive summary of the main points and key information in this document."
            with col1b:
                if st.button("🔍 Key Points", use_container_width=True):
                    st.session_state.auto_query = "Extract and list the key information, important facts, and main topics from this document."
            with col1c:
                if st.button("❓ Suggest Qs", use_container_width=True):
                    st.session_state.auto_query = "Based on this document, suggest 5 relevant questions that someone might want to ask about its content."
            
            # Query input        
            query = st.text_area(
                "Your question:",
                placeholder="What would you like to know about this document?",
                height=100,
                key="query_input"
            )
            if st.session_state.auto_query and not st.session_state.query:
                query=st.session_state.auto_query
                answer = get_answer(query)
                # Add to chat history
                st.session_state.Chat_history.append({
                    "question": query,
                    "answer": answer,
                    "type": "user"
                })                    
                
            ask_button = st.button("🚀 Get Answer", type="primary", use_container_width=True)
            
            if ask_button and query :
                st.session_state.query = query
                answer = get_answer(query)
                
                # Add to chat history
                st.session_state.Chat_history.append({
                    "question": query,
                    "answer": answer,
                    "type": "user"
                })
                # Rerun to update the chat history display
            if st.session_state.auto_query:
                st.session_state.auto_query=''    
                st.rerun()
        with col2:
            st.markdown('<div class="sub-header">📝 Conversation Area</div>', unsafe_allow_html=True)
            
            if not st.session_state.Chat_history:
                st.info("💡 No questions asked yet. Your conversation will appear here.")
            else:
                # Append all messages (stay inside chat box)
                for chat in st.session_state.Chat_history:
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>❓ Your Question:</strong><br>{chat['question']}
                        </div>
                        <div class="chat-message assistant-message">
                            <strong>🤖 Assistant Answer:</strong><br>{chat['answer']}
                        </div>
                    """, unsafe_allow_html=True)
            # Clear conversation button
            if st.session_state.Chat_history:
                if st.button("🗑️ Clear Conversation", use_container_width=True):
                    with st.spinner("Clearing history...."):    
                        clear_chat_history()
                        st.rerun()
         # Document analysis features
        st.markdown("---")
        st.markdown('<div class="sub-header">🔧 Document Analysis Tools</div>', unsafe_allow_html=True)
        col3,col4,col5 =st.columns(3)
        with col3:
            if st.button("📊 Extract Structured Data", use_container_width=True):
                    str_query="Extract any structured data, tables, lists, or organized information from this document and present it in a clear format."
                    answer=get_answer(str_query)
                    st.info(f"**Structured Data Found:**\n\n{answer}")
        with col4:
            if st.button("📈 Find Statistics", use_container_width=True):
                    stat_query="Find and list any statistical data, numbers, percentages, or quantitative information mentioned in this document."
                    answer = get_answer(stat_query)
                    st.info(f"**Statistical Information:**\n\n{answer}")
        with col5:
            if st.button("👥 Find People/Names", use_container_width=True):
                    names_query="List all the people's names, authors, important figures, or individuals mentioned in this document along with their context."
                    answer=get_answer(names_query)
                    st.info(f"**People Mentioned:**\n\n{answer}")
        st.markdown("---")
        with st.expander("📋 Current Document Details", expanded=False):
            col6, col7 = st.columns([2,1])
            with col6:
                st.metric("Document Name", st.session_state.document_name)
                st.metric("Document Type", st.session_state.document_type)
            with col7:
                st.metric("Total Characters", f"{len(st.session_state.info):,}")
                st.metric("Total Words", f"{len(st.session_state.info.split()):,}")
                    
                                             
if __name__ == '__main__':
    main()

