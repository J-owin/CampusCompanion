import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from backend.graph import app
import backend.granite
st.set_page_config(page_title="CampusCompanion")

st.sidebar.title("CampusCompanion")

st.sidebar.write("""
Ask me about:

📚 Academics

🏫 Campus

📅 Events

💰 Fees

📝 Admissions

🎓 Student Life
""")
st.info(
    "Welcome! I'm CampusCompanion. Ask me anything about your college."
)
question = st.text_input(
    "Ask a question",
    placeholder="e.g. What is the attendance requirement?"
)

if st.button("Ask"):
    if question:
        with st.spinner("Thinking..."):
            
            response = app.invoke({
            "question": question
            })
            print("Returning dictionary...")

        st.markdown("### CampusCompanion")

        st.success(response["answer"])

        st.markdown("### 📚 Sources")

        seen = set()

        for doc in response["sources"]:
            source = f"{Path(doc.metadata['source']).name} | Page {doc.metadata['page'] + 1}"

            if source not in seen:
                seen.add(source)
                st.write(f"📄 {source}")