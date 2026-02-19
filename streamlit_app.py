import streamlit as st
import tempfile
from speed_pipeline import process_video

st.set_page_config(page_title="Speed Detection AI", layout="wide")

st.title("🚗 AI Speed Violation Detection")

uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4","mov","avi"])

if uploaded_file:

    st.info("Processing video... please wait ⏳")

    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Run detection
    output_video = process_video(tfile.name)

    st.success("Processing Completed!")

    st.video(output_video)

    with open(output_video, "rb") as f:
        st.download_button("⬇ Download Result", f, file_name="speed_output.mp4")