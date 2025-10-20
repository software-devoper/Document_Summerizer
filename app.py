import streamlit as st
import pandas as pd
import tempfile
import os
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize the model
@st.cache_resource
def load_model():
    return ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        temperature=0
    )

def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False
    if 'document_text' not in st.session_state:
        st.session_state.document_text = ""
    if 'document_name' not in st.session_state:
        st.session_state.document_name = ""
    if 'document_type' not in st.session_state:
        st.session_state.document_type = ""

def load_document(file):
    """Load document based on file type"""
    try:
        file_extension = os.path.splitext(file.name)[1].lower()
        st.session_state.document_name = file.name
        st.session_state.document_type = file_extension.upper().replace('.', '')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        if file_extension == '.pdf':
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            content = "\n".join([doc.page_content for doc in docs])
            
        elif file_extension == '.txt':
            loader = TextLoader(tmp_file_path, encoding='utf-8')
            docs = loader.load()
            content = "\n".join([doc.page_content for doc in docs])
            
        elif file_extension == '.csv':
            # Use pandas to read CSV and convert to text
            df = pd.read_csv(tmp_file_path)
            content = df.to_string()
            
        else:
            return None, "Unsupported file format"
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return content, f"✅ Successfully loaded {file.name}"
    
    except Exception as e:
        return None, f"❌ Error loading file: {str(e)}"

def get_answer(query):
    """Get answer for the query using the document content"""
    try:
        model = load_model()
        
        pdf_prompt = PromptTemplate(
            input_variables=["user_query", "pdf_content"],
            template="""
You are an intelligent assistant that reads and understands documents.

Below is the extracted content from a document:

---------------------
{pdf_content}
---------------------

Based on the above content, answer the following user query clearly and accurately:

User Query: {user_query}

Important Instructions:
- Answer based ONLY on the provided document content
- If the answer cannot be found in the document, reply: "The answer is not available in the document."
- Be specific and provide relevant details from the document
- Format your response in a clear, readable way

Answer:
            """
        )

        parser = StrOutputParser()
        chain = pdf_prompt | model | parser
        
        response = chain.invoke({
            'pdf_content': st.session_state.document_text, 
            'user_query': query
        })
        
        return response
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"

def main():
    st.set_page_config(
        page_title="Smart Document Q&A Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Custom CSS for better styling
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
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #292d2f;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #292d2f;
        border-left: 4px solid #9c27b0;
    }
    .document-info {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
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
        
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'txt', 'csv'],
            help="Upload PDF, TXT, or CSV files to ask questions about their content"
        )
        
        if uploaded_file is not None:
            with st.spinner("Loading document..."):
                content, message = load_document(uploaded_file)
                
                if content:
                    st.session_state.document_text = content
                    st.session_state.document_loaded = True
                    st.success(message)
                    
                    # Document statistics
                    st.markdown("---")
                    st.subheader("📊 Document Information")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Characters", f"{len(content):,}")
                    with col2:
                        st.metric("Words", f"{len(content.split()):,}")
                    
                    st.info(f"**File Type:** {st.session_state.document_type}")
                    st.info(f"**File Name:** {st.session_state.document_name}")
                        
                    # Show document preview
                    with st.expander("📄 Document Preview", expanded=False):
                        preview_text = content[:800] + "..." if len(content) > 800 else content
                        st.text_area("Preview", preview_text, height=200, label_visibility="collapsed")
                else:
                    st.error(message)
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
        # Welcome page when no document is loaded
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
                value=st.session_state.get('auto_query', ''),
                placeholder="What would you like to know about this document?",
                height=100,
                key="query_input"
            )
            
            # Clear the auto_query after use
            if 'auto_query' in st.session_state and st.session_state.auto_query:
                st.session_state.auto_query = ""
            
            ask_col1, ask_col2, ask_col3 = st.columns([1, 2, 1])
            with ask_col2:
                ask_button = st.button("🚀 Get Answer", type="primary", use_container_width=True)
            
            if ask_button and query:
                with st.spinner("🔍 Analyzing document and generating answer..."):
                    answer = get_answer(query)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": answer,
                        "type": "user"
                    })
                    
                    # Rerun to update the chat history display
                    st.rerun()
    
        with col2:
            st.markdown('<div class="sub-header">📝 Conversation History</div>', unsafe_allow_html=True)
            
            if not st.session_state.chat_history:
                st.info("💡 No questions asked yet. Your conversation will appear here.")
            else:
                # Display chat history in reverse order (newest first)
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    if chat["type"] == "user":
                        with st.container():
                            st.markdown(f"""
                            <div class="chat-message user-message">
                                <strong>❓ Your Question:</strong><br>
                                {chat['question']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="chat-message assistant-message">
                                <strong>🤖 Assistant Answer:</strong><br>
                                {chat['answer']}
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("---")
            
            # Clear chat history button
            if st.session_state.chat_history:
                if st.button("🗑️ Clear Conversation History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
        
        # Document analysis features
        st.markdown("---")
        st.markdown('<div class="sub-header">🔧 Document Analysis Tools</div>', unsafe_allow_html=True)
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            if st.button("📊 Extract Structured Data", use_container_width=True):
                with st.spinner("Extracting structured data..."):
                    structured_query = "Extract any structured data, tables, lists, or organized information from this document and present it in a clear format."
                    answer = get_answer(structured_query)
                    st.info(f"**Structured Data Found:**\n\n{answer}")
        
        with col4:
            if st.button("📈 Find Statistics", use_container_width=True):
                with st.spinner("Looking for statistical data..."):
                    stats_query = "Find and list any statistical data, numbers, percentages, or quantitative information mentioned in this document."
                    answer = get_answer(stats_query)
                    st.info(f"**Statistical Information:**\n\n{answer}")
        
        with col5:
            if st.button("👥 Find People/Names", use_container_width=True):
                with st.spinner("Extracting names and people..."):
                    names_query = "List all the people's names, authors, important figures, or individuals mentioned in this document along with their context."
                    answer = get_answer(names_query)
                    st.info(f"**People Mentioned:**\n\n{answer}")

        # Current document info
        st.markdown("---")
        with st.expander("📋 Current Document Details", expanded=False):
            col6, col7 = st.columns(2)
            with col6:
                st.metric("Document Name", st.session_state.document_name)
                st.metric("Document Type", st.session_state.document_type)
            with col7:
                st.metric("Total Characters", f"{len(st.session_state.document_text):,}")
                st.metric("Total Words", f"{len(st.session_state.document_text.split()):,}")

if __name__ == "__main__":


    main()
