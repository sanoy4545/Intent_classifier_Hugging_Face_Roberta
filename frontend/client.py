import streamlit as st
import requests
import io
import zipfile

st.set_page_config(page_title="Conversation Intent Classifier", page_icon="🤖", layout="centered")

st.title("🤖 Multi-Turn Intent Classifier")
st.markdown("Upload a **JSON file** of conversations to classify intents and download results as a ZIP file.")

# Backend FastAPI endpoint (adjust if running elsewhere)
API_URL = "http://localhost:8000/classify"  # make sure FastAPI runs on this

uploaded_file = st.file_uploader("Upload your conversation JSON file", type=["json"])

if uploaded_file is not None:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    if st.button("Start Classification"):
        with st.spinner("Processing... Please wait ⏳"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/json")}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    st.success("Classification completed successfully 🎉")

                    # Receive ZIP file as bytes
                    zip_bytes = io.BytesIO(response.content)
                    st.download_button(
                        label="📦 Download Results (ZIP)",
                        data=zip_bytes,
                        file_name="classified_results.zip",
                        mime="application/zip"
                    )
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
