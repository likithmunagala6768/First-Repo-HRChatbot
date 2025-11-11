import streamlit as st 
import pandas as pd 
import fitz  # PyMuPDF for PDF text extraction 
import plotly.express as px 
from groq import Groq 
import io 
import json 

st.set_page_config(page_title="📊 Financial Chatbot", layout="wide")

st.title("📈 Financial Insights Chatbot")
st.caption("Upload PDFs or CSVs → Ask questions → Get insights + charts (powered by Groq Llama 3)")

# Replace with your actual API key 
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# ------------------------------
# FILE UPLOAD
# ------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF or CSV files",
    type=["pdf", "csv"],
    accept_multiple_files=True
)

all_text = ""
dataframes = []

if uploaded_files:
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            pdf = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page in pdf:
                text += page.get_text("text")
            all_text += f"\n\n### From {file.name}:\n{text}"

        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            dataframes.append(df)
            st.write(f"📄 **Preview of {file.name}:**")
            st.dataframe(df.head())
            all_text += f"\n\n### From {file.name}:\n{df.to_string(index=False)}"

if not all_text:
    st.info("👆 Please upload at least one PDF or CSV file to begin.")
    st.stop()

# ------------------------------
# USER QUESTION INPUT
# ------------------------------
question = st.text_input("💬 Ask a question about your data:")

# ------------------------------
# HELPER FUNCTION: VISUALIZE DATA
# ------------------------------
def plot_from_dataframe(df, question):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 1:
        st.warning("No numeric data found for visualization.")
        return

    question_lower = question.lower()

    # Decide visualization type based on keywords
    if "trend" in question_lower or "growth" in question_lower or "over time" in question_lower:
        fig = px.line(df, x=df.columns[0], y=numeric_cols, markers=True, title="📈 Trend / Growth Chart")
    elif "compare" in question_lower or "comparison" in question_lower or "difference" in question_lower:
        fig = px.bar(df, x=df.columns[0], y=numeric_cols, barmode="group", title="📊 Comparison Chart")
    elif "distribution" in question_lower or "share" in question_lower or "ratio" in question_lower or "percentage" in question_lower:
        if len(numeric_cols) >= 1:
            fig = px.pie(df, names=df.columns[0], values=numeric_cols[0], title="🥧 Distribution / Share Chart")
        else:
            st.warning("Need at least one numeric column for pie chart.")
            return
    elif "relationship" in question_lower or "correlation" in question_lower or "scatter" in question_lower:
        if len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title="📉 Relationship / Scatter Plot")
        else:
            fig = px.scatter(df, x=df.columns[0], y=numeric_cols[0], title="📉 Scatter Plot")
    elif "area" in question_lower:
        fig = px.area(df, x=df.columns[0], y=numeric_cols, title="📊 Area Chart")
    else:
        # Default fallback visualization
        fig = px.line(df, x=df.columns[0], y=numeric_cols, markers=True, title="📊 Default Line Chart")

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# MAIN LLM CALL
# ------------------------------
if question:
    with st.spinner("Analyzing your files..."):
        context = all_text[:12000]
        prompt = f"""
You are a financial analysis assistant.
You are given data (in text, CSV, or extracted tables) and a user question.
1. Analyze the data carefully.
2. Provide a concise, analytical answer.
3. If relevant, return a JSON summary of key metrics for visualization.
4. Suggest the most suitable chart type in the JSON output as `"chart_type"` (e.g., "bar", "line", "pie", "scatter", "area").

Example JSON format:
{{
  "chart_type": "bar",
  "Years": [2021, 2022, 2023],
  "Revenue": [100, 150, 200],
  "Profit": [20, 30, 45]
}}

Context:
{context}

Question:
{question}
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer = response.choices[0].message.content
        except Exception as e:
            st.error(f"Error from Groq API: {e}")
            st.stop()

    st.subheader("🧠 Llama 3 Answer:")
    st.write(answer)

    # ------------------------------
    # Attempt to parse and visualize JSON
    # ------------------------------
    try:
        json_start = answer.find("{")
        json_end = answer.rfind("}")
        if json_start != -1 and json_end != -1:
            json_str = answer[json_start:json_end+1]
            data = json.loads(json_str)

            chart_type = data.pop("chart_type", "line").lower()
            df = pd.DataFrame(data)

            st.subheader("📊 Visualization (from AI-generated data):")
            st.dataframe(df)

            # Render visualization based on chart_type
            if chart_type == "bar":
                fig = px.bar(df, x=df.columns[0], y=df.columns[1:], barmode="group", title="📊 Bar Chart")
            elif chart_type == "pie":
                fig = px.pie(df, names=df.columns[0], values=df.columns[1], title="🥧 Pie Chart")
            elif chart_type == "scatter":
                fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title="📉 Scatter Plot")
            elif chart_type == "area":
                fig = px.area(df, x=df.columns[0], y=df.columns[1:], title="🌈 Area Chart")
            else:
                fig = px.line(df, x=df.columns[0], y=df.columns[1:], markers=True, title="📈 Line Chart")

            st.plotly_chart(fig, use_container_width=True)
        else:
            if dataframes:
                st.subheader("📊 Visualization (based on CSV):")
                plot_from_dataframe(dataframes[0], question)

    except Exception as e:
        st.warning(f"Could not extract structured data for visualization. ({e})")
        if dataframes:
            plot_from_dataframe(dataframes[0], question)
