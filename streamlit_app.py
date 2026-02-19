import streamlit as st
import tempfile
import os
from speed_pipeline import process_video

st.set_page_config(page_title="Vehicle Speed Detection", layout="wide")

st.title("🚗 Vehicle Speed Detection System")
st.write("Upload a traffic video to detect vehicle speed and violations.")

uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4","avi","mov"])

if uploaded_video is not None:

    st.success("Video uploaded successfully!")

    # Save uploaded video temporarily
    temp_input = tempfile.NamedTemporaryFile(delete=False)
    temp_input.write(uploaded_video.read())
    input_path = temp_input.name

    st.info("⏳ Processing video... please wait")

    # Run detection pipeline
    output_video, df = process_video(input_path)

    st.success("Processing Completed!")

    # SHOW VIDEO
    st.subheader("🎥 Processed Video")
    video_file = open(output_video, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    # DOWNLOAD BUTTON
    st.download_button(
        label="⬇ Download Processed Video",
        data=video_bytes,
        file_name="processed_video.mp4",
        mime="video/mp4"
    )

    # SHOW TABLE
    st.subheader("🚨 Violations Detected")

    if len(df) > 0:
        st.dataframe(df, use_container_width=True)
    else:
        st.success("No violations detected 🎉")

    os.remove(input_path)
