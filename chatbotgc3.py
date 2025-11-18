import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import plotly.express as px
from groq import Groq
import json
import os
 
# -------------------------------
# ⚙️ Page Config
# -------------------------------
st.set_page_config(page_title="📊 Financial Chatbot", layout="wide")
 
# -------------------------------
# 🌐 Groq API Configuration
# -------------------------------
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)
 
# -------------------------------
# SESSION STATE INIT
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
 
if "latest_answer" not in st.session_state:
    st.session_state.latest_answer = ""
 
if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = ""
 
if "dataframes" not in st.session_state:
    st.session_state.dataframes = []
 
if "selected_question_index" not in st.session_state:
    st.session_state.selected_question_index = None
 
 
# -------------------------------
# Helper: Fix DataFrame for Streamlit Arrow Compatibility
# -------------------------------
def fix_arrow_df(df: pd.DataFrame):
    """Convert object/mixed columns to string to avoid Arrow serialization errors."""
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    return df
 
 
# -------------------------------
# SIDEBAR: Chat History
# -------------------------------
st.sidebar.header("Chat History")
 
for i, entry in enumerate(st.session_state.chat_history):
    if st.sidebar.button(f"Q{i+1}: {entry['question']}", key=f"btn_{i}"):
        st.session_state.selected_question_index = i
 
if st.session_state.selected_question_index is not None:
    selected = st.session_state.chat_history[st.session_state.selected_question_index]
    st.sidebar.markdown("### Answer:")
    st.sidebar.write(selected["answer"])
 
 
# -------------------------------
# 📂 Load Files from /uploads Directory
# -------------------------------
UPLOAD_DIR = "uploads"
 
if not os.path.exists(UPLOAD_DIR):
    st.error("Uploads directory not found. Create an 'uploads' folder.")
    st.stop()
 
existing_files = os.listdir(UPLOAD_DIR)
if not existing_files:
    st.error("No files found in uploads folder.")
    st.stop()
 
all_text = ""
dfs = []
 
for file_name in existing_files:
    file_path = os.path.join(UPLOAD_DIR, file_name)
 
    if file_name.endswith(".pdf"):
        pdf = fitz.open(file_path)
        text = "".join(page.get_text("text") for page in pdf)
        all_text += f"\n\n### From {file_name}:\n{text}"
 
    elif file_name.endswith(".csv"):
        df = pd.read_csv(file_path)
        dfs.append(df)
        all_text += f"\n\n### From {file_name}:\n{df.to_string(index=False)}"
 
st.session_state.uploaded_text = all_text
st.session_state.dataframes = dfs
 
 
# -------------------------------
# 🎨 Chat Interface Styling
# -------------------------------
st.markdown("""
<style>
.chat-bubble-user, .chat-bubble-ai {
font-family: "Segoe UI", Arial, sans-serif;
font-size: 16px;
line-height: 1.5;
white-space: pre-wrap;
word-wrap: break-word;
}
.chat-bubble-user {
background-color: #0078ff;
color: white;
padding: 10px 14px;
border-radius: 12px;
margin: 5px 0px;
width: fit-content;
max-width: 80%;
align-self: flex-end;
}
.chat-bubble-ai {
background-color: #f1f1f1;
color: #333;
padding: 10px 14px;
border-radius: 12px;
margin: 5px 0px;
width: fit-content;
max-width: 80%;
align-self: flex-start;
}
.chat-container {
display: flex;
flex-direction: column;
}
</style>
""", unsafe_allow_html=True)
 
st.title("📈 Financial Insights Chatbot")
st.caption("Ask questions → Get insights + visualizations (Groq Llama 3)")
 
 
# -------------------------------
# 💬 Question Input
# -------------------------------
question = st.chat_input("💬 Ask a question about your data...")
 
if question and (not st.session_state.chat_history or
                 st.session_state.chat_history[-1]["question"] != question):
 
    if not st.session_state.uploaded_text:
        st.warning("No data available in uploads folder.")
        st.stop()
 
    with st.spinner("Analyzing your files..."):
 
        context = st.session_state.uploaded_text[:12000]
 
        prompt = f"""
You are a financial analysis assistant.
You ONLY use the following extracted file data for answering.
If the answer is not in the data, say:
"The information is not available in the uploaded files."
 
Answer format:
<answer>
[clean explanation]
</answer>
 
For visualization, include:
<json>
{{ "chart_type": "bar/line/pie", "col1": [...], ... }}
</json>
 
DATA:
{context}
 
QUESTION:
{question}
"""
 
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Strictly use uploaded files only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
 
            raw_answer = response.choices[0].message.content
 
            # Extract clean answer
            a_start = raw_answer.find("<answer>")
            a_end = raw_answer.find("</answer>")
            clean_answer = raw_answer[a_start+8:a_end].strip() if a_start != -1 else raw_answer
 
        except Exception as e:
            st.error(f"Groq API error: {e}")
            st.stop()
 
        st.session_state.chat_history.append({
            "question": question,
            "answer": clean_answer,
            "raw_answer": raw_answer
        })
 
 
# -------------------------------
# 💬 Chat Display
# -------------------------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
 
for chat in st.session_state.chat_history:
    st.markdown(f"<div class='chat-bubble-user'>{chat['question']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble-ai'>{chat['answer']}</div>", unsafe_allow_html=True)
 
st.markdown("</div>", unsafe_allow_html=True)
 
 
# -------------------------------
# 📊 Visualization
# -------------------------------
if st.session_state.chat_history:
 
    latest_raw = st.session_state.chat_history[-1]["raw_answer"]
 
    json_start = latest_raw.find("<json>")
    json_end = latest_raw.find("</json>")
 
    if json_start != -1 and json_end != -1:
        try:
            json_str = latest_raw[json_start + 6:json_end].strip()
            data = json.loads(json_str)
 
            chart_type = data.pop("chart_type", "line").lower()
 
            df = pd.DataFrame(data)
            df = fix_arrow_df(df)   # 🔥 Prevent Arrow crash
 
            st.subheader("📊 Visualization")
            st.dataframe(df)
 
            # ----- Plot Selection -----
            if chart_type == "bar":
                fig = px.bar(df, x=df.columns[0], y=df.columns[1:], barmode="group")
 
            elif chart_type == "pie":
                fig = px.pie(df, names=df.columns[0], values=df.columns[1])
 
            elif chart_type == "scatter":
                fig = px.scatter(df, x=df.columns[0], y=df.columns[1])
 
            elif chart_type == "area":
                fig = px.area(df, x=df.columns[0], y=df.columns[1:])
 
            else:
                fig = px.line(df, x=df.columns[0], y=df.columns[1:], markers=True)
 
            st.plotly_chart(fig, use_container_width=True)
 
        except Exception as e:
            st.warning(f"Invalid JSON structure. Error: {e}")
 
    else:
        # CSV fallback
        if st.session_state.dataframes:
 
            df = st.session_state.dataframes[0]
            df = fix_arrow_df(df)
 
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
 
            if numeric_cols:
                st.subheader("📊 CSV Data Visualization")
                fig = px.line(df, x=df.columns[0], y=numeric_cols, markers=True)
                st.plotly_chart(fig, use_container_width=True)
 
 