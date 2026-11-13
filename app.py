
import streamlit as st
from google import genai
from dotenv import load_dotenv
import os
from PIL import Image
from io import BytesIO

load_dotenv()

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ AI Image Classification System")
st.caption("Powered by Gemini Vision AI")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API key missing. Check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

st.sidebar.title("About Project")
st.sidebar.write("AI-powered image understanding system.")
st.sidebar.write("Upload an image to identify its contents.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("AI Analysis")

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            """
                            Analyze this image and provide:
                            1. Main category
                            2. Main objects visible
                            3. Detailed description
                            4. Possible scene or activity
                            5. Important visual details

                            Clearly separate each section.
                            Only describe what is visibly supported.
                            """,
                            image
                        ]
                    )

                    result = response.text

                    st.session_state["analysis"] = result
                    st.session_state["image_name"] = uploaded_file.name

                except Exception as e:
                    st.error(f"Analysis failed: {e}")

if "analysis" in st.session_state:
    st.divider()
    st.subheader("📊 Classification Results")

    st.markdown(st.session_state["analysis"])

    report = (
        "AI IMAGE CLASSIFICATION REPORT\n\n"
        + st.session_state["analysis"]
    )

    st.download_button(
        "Download Analysis Report",
        data=report,
        file_name="image_analysis.txt",
        mime="text/plain"
    )
