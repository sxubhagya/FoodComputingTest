import streamlit as st
import cv2
import numpy as np
from PIL import Image as PILImage
from scan_and_lookup import detect_and_decode_barcode
import os
import json

# Convert secrets AttrDict to regular dict before dumping
with open("/tmp/gcp_key.json", "w") as f:
    json.dump(dict(st.secrets["gcp_service_account"]), f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp_key.json"


st.title("Food Product Recognition")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = PILImage.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    temp_path = "temp_uploaded_image.png"
    image.save(temp_path)
    # Show spinner while processing
    with st.spinner("Processing your image..."):
        result = detect_and_decode_barcode(image_cv, temp_path)
    st.subheader("Most Relevant Result")
    if result:
        for k, v in result.items():
            st.write(f"**{k}:** {v}")
    else:
        st.write("No product found.")