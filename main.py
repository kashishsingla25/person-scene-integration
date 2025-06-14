import streamlit as st
import cv2
import numpy as np
from utils import extract_person, blend_images

st.title("Person-Scene Integration")

# Upload and display person image
person_image = st.file_uploader("Upload Person Image (Full Body Recommended)", type=["jpg", "png", "jpeg"])
if person_image is not None:
    st.image(person_image, caption="Uploaded Person Image", use_container_width=True)

# Upload and display background image
background_image = st.file_uploader("Upload Background Image", type=["jpg", "png", "jpeg"])
if background_image is not None:
    st.image(background_image, caption="Uploaded Background Image", use_container_width=True)

# Process and display composite image
if person_image and background_image:
    person_bytes = person_image.read()
    background_bytes = background_image.read()

    # Extract person with alpha channel
    try:
        person = extract_person(person_bytes)
    except Exception as e:
        st.error(f"Extraction failed: {str(e)}. Please ensure the person image contains a full body for best results.")
        st.stop()

    # Debug: Display the extracted person image
    st.image(person, caption="Extracted Person (Debug)", use_container_width=True)

    # Debug: Display the alpha mask
    mask = person[:, :, 3]
    st.image(mask, caption="Alpha Mask (Debug)", use_container_width=True, clamp=True)

    # Decode background image to NumPy array
    background = cv2.imdecode(np.frombuffer(background_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Blend images
    try:
        result = blend_images(person, background)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="Composite Image", use_container_width=True)
    except Exception as e:
        st.error(f"Blending failed: {str(e)}")
