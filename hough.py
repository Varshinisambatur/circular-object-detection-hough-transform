import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# --- App Configuration ---
st.set_page_config(page_title="Hough Circle Detector", layout="wide")

st.title("Circular Objects Detection Using Hough Transform")
st.markdown("""
This tool wraps the OpenCV **HoughCircles** algorithm. 
Upload an image, tune the parameters in the sidebar, and detect circular objects in real-time.
""")

# --- Sidebar Parameters ---
st.sidebar.header("🔧 Detection Parameters")

st.sidebar.info("**1. Pre-processing**")
blur_ksize = st.sidebar.slider(
    "Median Blur Kernel Size", 
    min_value=1, max_value=15, value=5, step=2,
    help="Reduces noise before detection. Must be an odd number. Higher values remove more noise but blur edges."
)

st.sidebar.info("**2. Hough Transform Settings**")

dp = st.sidebar.slider(
    "DP (Inverse Ratio)", 
    min_value=1.0, max_value=5.0, value=1.2, step=0.1,
    help="Inverse ratio of the accumulator resolution to the image resolution. 1 = same resolution, 2 = half resolution."
)

min_dist = st.sidebar.slider(
    "Min Distance (pixels)", 
    min_value=10, max_value=500, value=50, step=10,
    help="Minimum distance between the centers of the detected circles. If too small, multiple neighbor circles may be falsely detected."
)

param1 = st.sidebar.slider(
    "Param 1 (Canny Threshold)", 
    min_value=10, max_value=300, value=50, step=10,
    help="Higher threshold for the internal Canny edge detector."
)

param2 = st.sidebar.slider(
    "Param 2 (Accumulator Threshold)", 
    min_value=5, max_value=200, value=30, step=1,
    help="The smaller it is, the more false circles may be detected. Corresponds to the number of votes needed to declare a center."
)

st.sidebar.info("**3. Size Constraints**")
min_radius = st.sidebar.slider("Min Radius", 0, 200, 10)
max_radius = st.sidebar.slider("Max Radius", 0, 500, 100)

# --- Main Logic ---

uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    # 1. Open with PIL
    pil_image = Image.open(uploaded_file)
    
    # 2. Convert to NumPy array
    img_array = np.array(pil_image)
    
    # 3. Handle Grayscale vs RGB uploads
    if len(img_array.shape) == 2:
        # Image is already grayscale
        original_color = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        gray = img_array
    else:
        # Image is RGB (Streamlit uses RGB, OpenCV usually uses BGR)
        # We keep 'img_array' as RGB for display purposes
        original_color = img_array
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # --- Processing ---
    
    # 1. Blur (Essential for Hough to avoid false positives from noise)
    # Ensure kernel size is odd
    if blur_ksize % 2 == 0: blur_ksize += 1
    gray_blurred = cv2.medianBlur(gray, blur_ksize)

    # 2. Hough Circle Transform
    # Note: We assume method is cv2.HOUGH_GRADIENT
    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=dp, 
        minDist=min_dist,
        param1=param1, 
        param2=param2,
        minRadius=min_radius, 
        maxRadius=max_radius
    )

    # 3. Visualization
    output_img = original_color.copy()
    circle_count = 0
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle_count = circles.shape[1]
        
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(output_img, (i[0], i[1]), i[2], (0, 255, 0), 3)
            # draw the center of the circle
            cv2.circle(output_img, (i[0], i[1]), 2, (255, 0, 0), 5)

    # --- Display Results ---
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original / Pre-processed")
        st.image(original_color, caption="Original Image", use_column_width=True)
        with st.expander("See Blurred Input (Debugging)"):
            st.image(gray_blurred, caption=f"Median Blur (k={blur_ksize})", use_column_width=True)

    with col2:
        st.subheader("Detection Result")
        st.image(output_img, caption=f"Found {circle_count} Circles", use_column_width=True)
        
        if circle_count > 0:
            st.success(f"✅ Successfully detected {circle_count} circular objects.")
            # Optional: Show data frame of coordinates
            with st.expander("View Circle Coordinates"):
                st.dataframe(pd.DataFrame(circles[0], columns=["X", "Y", "Radius"]))
        else:
            st.warning("⚠️ No circles found. Try lowering 'Param 2' or adjusting 'Min/Max Radius'.")

else:
    st.info("Please upload an image to begin.")