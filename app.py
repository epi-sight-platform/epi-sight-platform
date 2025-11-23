import streamlit as st
import pandas as pd
import plotly.express as px
from pypdf import PdfReader
import io

# --- GOOGLE GENAI CLIENT IMPORT ---
try:
    from google import genai
    # Define the model to use
    MODEL = "gemini-2.5-flash" 
except ImportError:
    st.error("🚨 Required library 'google-genai' not found. Please ensure it is installed.")
    st.stop()

# --- CONFIGURATION ---
st.set_page_config(page_title="EpiSight Platform", layout="wide", page_icon="🏥")

# Attempt to initialize Gemini Client using secrets
try:
    # REQUIRES: The GEMINI_API_KEY must be set in the Streamlit secrets.toml file.
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"🚨 Configuration Error: Gemini API Key not found or invalid. Please check Streamlit secrets.")
    st.stop() 

# --- CSS FOR WEBSITE FEEL ---
st.markdown("""
<style>
    /* Main background color */
    .main { background-color: #f5f5f5; }
    /* Primary headline color (Teal) */
    h1 { color: #006064; }
    /* Style for main buttons (Enhanced CSS for better look) */
    .stButton>button { 
        border-radius: 5px; 
        background-color: #008080; 
        color: white; 
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #004d4d;
    }
    /* Style for the sidebar navigation */
    .st-emotion-cache-p5mhrd {
        background-color: #e0f7fa;
        border-radius: 0 10px 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS (Cleaning, Reporting, PDF) ---

def extract_text_from_pdf(file):
    """Extracts all text content from a PDF file object."""
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            # Safely handle pages with no extractable text
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def smart_clean_data(df):
    """Applies an automated data cleaning protocol and returns a report."""
    report = []
    
    # 1. Clean Headers (Snake Case)
    old_cols = list(df.columns)
    df.columns = [
        c.strip().lower()
        .replace(' ', '_')
        .replace('-', '_')
        .replace('(', '') 
        .replace(')', '')
        for c in df.columns
    ]
    if list(df.columns) != old_cols:
        report.append("✅ Standardized column headers (snake_case, removed parentheses).")
        
    # 2. Remove Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        report.append(f"✅ Removed {duplicates} duplicate rows.")
    
    # 3. Handle Missing Values
    # Impute numeric columns with median
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report.append(f"✅ Imputed {missing_count} missing values in numeric column '{col}' with median ({median_val}).")

    # Impute categorical columns with 'Unknown'
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna("Unknown")
            report.append(f"✅ Imputed {missing_count} missing values in categorical column '{col}' with 'Unknown'.")
        
    return df, report

def generate_canva_report(title, analysis_text, chart_html=None):
    """Generates an HTML report styled professionally (Canva-style)."""
    html_template = f"""
    <html>
    <head>
        <title>{title}</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Roboto', sans-serif; background-color: #f4f6f8; color: #333; margin: 0; padding: 40px; }}
            .container {{ max-width: 900px; margin: auto; }}
            .header {{ 
                background: linear-gradient(135deg, #006064 0%, #00acc1 100%); 
                color: white; 
                padding: 40px; 
                border-radius: 15px; 
                box-shadow: 0 10px 20px rgba(0,0,0,0.15); 
                margin-bottom: 30px; 
            }}
            .header h1 {{ margin: 0; font-size: 2.5em; }}
            .card {{ 
                background: white; 
                padding: 30px; 
                border-radius: 12px; 
                box-shadow: 0 4px 12px rgba(0,0,0,0.08); 
                margin-bottom: 25px; 
            }}
            .card h2 {{ color: #006064; border-bottom: 2px solid #e0f7fa; padding-bottom: 10px; margin-top: 0; }}
            .analysis-text {{ line-height: 1.6; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h1>{title}</h1></div>
            <div class="card">
                <h2>Key Findings & Summary </h2>
                <div class="analysis-text">{analysis_text.replace(chr(10), '<br>')}</div>
            </div>
            <div class="card">
                <h2>Data Visualization</h2>
                <div style="width:100%; overflow:hidden;">
                    {chart_html if chart_html else "<p>No suitable chart generated for this report.</p>"}
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_template


# --- GLOBAL STATE & NAVIGATION ---
if "data" not in st.session_state:
    st.session_state["data"] = None
if "raw_text" not in st.session_state:
    st.session_state["raw_text"] = None

if "system_instruction" not in st.session_state:
    st.session_state["system_instruction"] = (
        "You are EpiSight, an expert monitoring and evaluation (M&E) and public health data analyst. "
        "Your responses must be concise, professional, and directly address the user's request using "
        "the provided DATA CONTEXT or DOCUMENT CONTEXT. If no data is available, politely state that you need data to proceed. "
        "If you generate analysis, keep the output readable and formatted with markdown."
    )

if "messages" not in st.session_state:
    # Chat history only stores user/assistant turns for display
    st.session_state["messages"] = []


# Sidebar Navigation
st.sidebar.title("🏥 EpiSight Portal")
page = st.sidebar.radio("Navigate", ["📂 Data Ingestion", "🧹 Data Cleaning", "📊 Analysis & Chat", "📝 Report Designer"])

# --- PAGE LOGIC ---

# 1. DATA INGESTION TAB
if page == "📂 Data Ingestion":
    st.title("Data Ingestion & Setup")
    st.markdown("Upload your raw M&E data (CSV, Excel, or PDF) for immediate analysis.")

    uploaded_file = st.file_uploader("Drop files here", type=['csv', 'xlsx', 'pdf'], key="data_uploader")
    
    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1]
        
        # Reset data states before loading new file
        st.session_state["data"] = None
        st.session_state["raw_text"] = None

        try:
            if file_ext == "csv":
                st.session_state["data"] = pd.read_csv(uploaded_file)
                st.success(f"Loaded CSV: {len(st.session_state['data'])} records.")
            elif file_ext == "xlsx":
                st.session_state["data"] = pd.read_excel(uploaded_file)
                st.success(f"Loaded Excel: {len(st.session_state['data'])} records.")
            elif file_ext == "pdf":
                text_data = extract_text_from_pdf(uploaded_file)
                if text_data:
                    st.session_state["raw_text"] = text_data
                    st.success("PDF Text Extracted. Use the 'Analysis & Chat' tab to query the document content.")
                else:
                    st.error("Could not extract meaningful text from the PDF.")
        except Exception as e:
            st.error(f"Failed to load file: {e}")
        
        if st.session_state["data"] is not None:
             st.subheader("Data Preview (First 5 Rows)")
             st.dataframe(st.session_state["data"].head())


# 2. DATA CLEANING TAB
elif page == "🧹 Data Cleaning":
    st.title("Smart Data Cleaning Protocol")
    
    if st.session_state["data"] is not None:
        st.subheader(f"Current Data Status ({len(st.session_state['data'])} rows)")
        
        # Display summary statistics before cleaning
        st.dataframe(st.session_state["data"].head())

        if st.button("✨ Run Auto-Clean Protocol", key="clean_button"):
            with st.spinner("Cleaning in progress..."):
                cleaned_df, clean_log = smart_clean_data(st.session_state["data"].copy())
                st.session_state["data"] = cleaned_df
                st.success("Data Cleaning Complete!")
                
                st.subheader("Cleaning Summary")
                for log_item in clean_log:
                    st.markdown(f"- {log_item}")
                st.info("The cleaned data is now active for analysis.")
                
                st.subheader("Data Summary After Cleaning")
                st.dataframe(st.session_state["data"].describe(include='all'))
        
    else:
        st.warning("Please upload a CSV or Excel file first in the 'Data Ingestion' tab to enable cleaning.")


# 3. ANALYSIS & CHAT TAB
elif page == "📊 Analysis & Chat":
    st.title("AI Analyst Chat")
    
    if st.session_state["data"] is None and st.session_state["raw_text"] is None:
        st.warning("Please upload data or a document in the 'Data Ingestion' tab to start analysis.")
    else:
        st.info("Ask any M&E or Public Health questions about your uploaded data/document.")

    # Display chat history 
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    if prompt := st.chat_input("Analyze trends, calculate prevalence, or ask for a report..."):
        
        # Display user input immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # 1. Prepare context for the AI
            data_context = ""
            if st.session_state["data"] is not None:
                df = st.session_state["data"]
                # Limit the sample data string size to prevent token overload
                sample_data_string = df.head(5).to_string()[:2000] 
                data_context = (
                    f"DATA CONTEXT: The dataset has {len(df)} rows. Columns: {list(df.columns)}. "
                    f"Data types: {df.dtypes.to_dict()}. Sample Data (First 5 Rows): {sample_data_string}."
                )
            elif st.session_state["raw_text"] is not None:
                # Truncate document context for API call efficiency
                doc_text = st.session_state['raw_text']
                data_context = f"DOCUMENT CONTEXT: The uploaded document text is: {doc_text[:4000]}..."

            # 2. Construct the full prompt with the context for the API call
            full_prompt_with_context = f"Data Analyst Request: {prompt}\n\n{data_context}"
            
            # 3. Construct the full message list for the API (mapping Streamlit roles to Gemini roles)
            gemini_contents = []
            
            # Add existing history
            for msg in st.session_state.messages:
                # 'assistant' role in Streamlit maps to 'model' role in Gemini
                role = "user" if msg["role"] == "user" else "model"
                gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
            
            # Add the current user message (with context)
            gemini_contents.append({"role": "user", "parts": [{"text": full_prompt_with_context}]})
            
            # 4. Call the Gemini API
            try:
                with st.spinner("EpiSight Analyst is processing..."):
                    
                    response = client.models.generate_content(
                        model=MODEL,
                        contents=gemini_contents,
                        config={
                            "system_instruction": st.session_state["system_instruction"]
                        }
                    )
                
                ai_reply = response.text
                message_placeholder.markdown(ai_reply)
                
                # 5. Update session history with clean prompts (NO data context)
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            except Exception as e:
                st.error(f"An error occurred during AI generation. Please check your API key and connection: {e}")


# 4. REPORT DESIGNER TAB
elif page == "📝 Report Designer":
    st.title("Canva-Style Report Generation")
    
    # Check if we have data and at least one AI response to base the report on
    # We only care about the assistant's generated text, not the system message.
    analysis_messages = [msg['content'] for msg in st.session_state.messages if msg['role'] == 'assistant']
    
    if st.session_state["data"] is not None and len(analysis_messages) > 0:
        st.info("Generating a report based on the most recent AI analysis of your data.")
        
        # Get the most recent assistant message for the summary text
        analysis_text = analysis_messages[-1] 
        
        st.subheader("Report Configuration")
        title = st.text_input("Report Title", "Monthly M&E Update")
        
        # --- Chart Configuration (Selectable Columns) ---
        chart_html = ""
        st.subheader("Chart Inclusion")
        
        df = st.session_state["data"]
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        chart_type = st.selectbox("Select Chart Type", ["None", "Bar Chart", "Scatter Plot", "Histogram"])

        if chart_type == "Bar Chart" and len(cat_cols) > 0:
            count_col = st.selectbox("Select Categorical Column for Counts:", cat_cols)
            if count_col:
                count_data = df[count_col].value_counts().reset_index()
                # Rename columns appropriately for plotly
                count_data.columns = [count_col, 'Count'] 
                fig = px.bar(count_data, x=count_col, y='Count', 
                             title=f"Count of Records by {count_col}", color='Count', template="plotly_white")
                chart_html = fig.to_html(full_html=False)
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter Plot" and len(num_cols) >= 2:
            x_col = st.selectbox("Select X-axis (Numeric)", num_cols, index=0)
            y_col = st.selectbox("Select Y-axis (Numeric)", num_cols, index=1 if len(num_cols) > 1 else 0)
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, 
                                 title=f"Correlation: {y_col} vs. {x_col}", template="plotly_white")
                chart_html = fig.to_html(full_html=False)
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Histogram" and len(num_cols) > 0:
            hist_col = st.selectbox("Select Numeric Column for Distribution:", num_cols)
            if hist_col:
                fig = px.histogram(df, x=hist_col, 
                                 title=f"Distribution of {hist_col}", template="plotly_white")
                chart_html = fig.to_html(full_html=False)
                st.plotly_chart(fig, use_container_width=True)

        
        if st.button("📄 Finalize & Download Report"):
            report_html = generate_canva_report(title, analysis_text, chart_html)
            
            st.download_button(
                label="📥 Download Designed Report (HTML)",
                data=report_html,
                file_name=f"{title.replace(' ', '_')}.html",
                mime="text/html"
            )
            st.balloons()
            
    else:
        st.warning("Please ensure you have uploaded data/document and run an analysis in the 'Analysis & Chat' tab before generating a report.")
