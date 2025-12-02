import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Interactive DIP Tool", layout="wide")

st.title("🖼️ Interactive Digital Image Processing Tool")
st.markdown("""
This tool allows you to apply various **Spatial Domain Operations** on images.
Choose between **Intensity Transformations** and **Neighborhood Operations**.
""")

# --- Sidebar: Image Upload ---
st.sidebar.header("1. Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tif"])

def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE) # Load as grayscale for DIP basics
    return img

if uploaded_file is not None:
    original_image = load_image(uploaded_file)
    
    if original_image is None:
        st.error("Error loading image. Please upload a valid image file.")
        st.stop()
    
    # Display Original Image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image (Grayscale)")
        st.image(original_image, use_container_width=True, clamp=True, channels='GRAY')
        
        # Histogram of original
        if st.checkbox("Show Original Histogram"):
            fig, ax = plt.subplots()
            ax.hist(original_image.ravel(), 256, [0, 256])
            st.pyplot(fig)

    # --- Sidebar: Operation Selection ---
    st.sidebar.header("2. Select Operation Type")
    operation_type = st.sidebar.radio(
        "Choose Domain:",
        ("Intensity (Point) Transformations", "Neighborhood (Spatial Filtering) Operations")
    )

    processed_image = original_image.copy()
    
    # ==========================================
    # 1. INTENSITY / POINT TRANSFORMATIONS
    # ==========================================
    if operation_type == "Intensity (Point) Transformations":
        st.sidebar.subheader("Intensity Transformations")
        transform_mode = st.sidebar.selectbox(
            "Select Transformation:",
            [
                "Image Negative",
                "Log Transformation",
                "Inverse Log Transformation",
                "Power-law (Gamma) Transform",
                "Contrast Stretching",
                "Gray-level Slicing",
                "Bit-plane Slicing"
            ]
        )

        if transform_mode == "Image Negative":
            # s = L - 1 - r
            processed_image = 255 - original_image
            st.sidebar.info("Formula: s = 255 - r")

        elif transform_mode == "Log Transformation":
            # s = c * log(1 + r)
            # Calculate default c safely
            img_max = np.max(original_image)
            if img_max > 0:
                default_c = 255 / np.log(1 + float(img_max))
            else:
                default_c = 1.0
            
            # Ensure slider range accommodates the default value
            max_c_slider = max(100.0, float(default_c) * 1.2)
            
            c = st.sidebar.slider("Constant (c)", 0.0, float(max_c_slider), float(default_c))
            
            processed_image = c * (np.log(1 + original_image.astype(np.float64)))
            processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
            st.sidebar.info("Formula: s = c * log(1 + r)")

        elif transform_mode == "Inverse Log Transformation":
            # s = c * (exp(r) - 1)
            # Normalize to 0-1 for stability then scale back
            c = st.sidebar.slider("Constant (c)", 0.0, 5.0, 1.0)
            # Approximation for visualization
            norm_img = original_image / 255.0
            processed_image = c * (np.exp(norm_img) - 1)
            # Scale to 0-255 and clip
            processed_image = np.clip(processed_image * 255, 0, 255).astype(np.uint8)
            st.sidebar.info("Formula: s = c * (exp(r) - 1)")

        elif transform_mode == "Power-law (Gamma) Transform":
            # s = c * r^gamma
            gamma = st.sidebar.slider("Gamma (γ)", 0.1, 5.0, 1.0, 0.1)
            c = st.sidebar.number_input("Constant (c)", value=1.0)
            
            # Normalize, apply gamma, scale back
            norm_img = original_image / 255.0
            processed_image = c * np.power(norm_img, gamma)
            # Scale to 0-255 and clip
            processed_image = np.clip(processed_image * 255, 0, 255).astype(np.uint8)
            
            st.sidebar.info(f"Formula: s = c * r^{gamma}")
            if gamma < 1:
                st.sidebar.text("Expands dark values (Brightens image)")
            elif gamma > 1:
                st.sidebar.text("Compresses dark values (Darkens image)")

        elif transform_mode == "Contrast Stretching":
            # Piecewise linear
            st.sidebar.text("Define points (r1, s1) and (r2, s2)")
            r1 = st.sidebar.slider("r1", 0, 255, 70)
            s1 = st.sidebar.slider("s1", 0, 255, 0)
            r2 = st.sidebar.slider("r2", 0, 255, 140)
            s2 = st.sidebar.slider("s2", 0, 255, 255)
            
            def pixel_val(pix, r1, s1, r2, s2):
                if 0 <= pix <= r1:
                    return (s1 / r1) * pix
                elif r1 < pix <= r2:
                    return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
                else:
                    return ((255 - s2) / (255 - r2)) * (pix - r2) + s2
            
            # Vectorize for speed
            pixel_val_vec = np.vectorize(pixel_val)
            # Handle division by zero edge cases in UI logic or simple try-except
            try:
                processed_image = pixel_val_vec(original_image, r1, s1, r2, s2)
                processed_image = processed_image.astype(np.uint8)
            except:
                st.error("Avoid r1=0 or r1=r2 for this simple implementation.")

        elif transform_mode == "Gray-level Slicing":
            min_r = st.sidebar.slider("Min Intensity", 0, 255, 100)
            max_r = st.sidebar.slider("Max Intensity", 0, 255, 200)
            background = st.sidebar.radio("Background:", ("Retain Background", "Make Black"))
            
            height, width = original_image.shape
            new_img = np.zeros((height, width), dtype=np.uint8)
            
            if background == "Retain Background":
                new_img = original_image.copy()
                # Highlight range
                mask = (original_image >= min_r) & (original_image <= max_r)
                new_img[mask] = 255
            else:
                # Black background
                mask = (original_image >= min_r) & (original_image <= max_r)
                new_img[mask] = 255
                
            processed_image = new_img

        elif transform_mode == "Bit-plane Slicing":
            plane = st.sidebar.slider("Bit Plane", 0, 7, 7)
            # Extract bit plane
            # Bitwise AND with 2^plane
            processed_image = cv2.bitwise_and(original_image, 2**plane)
            # Scale to 255 for visibility
            processed_image = processed_image * (255 // (2**plane)) # Or just threshold > 0
            st.sidebar.text(f"Showing Bit Plane {plane}")

    # ==========================================
    # 2. NEIGHBORHOOD / SPATIAL FILTERING
    # ==========================================
    elif operation_type == "Neighborhood (Spatial Filtering) Operations":
        st.sidebar.subheader("Spatial Filtering")
        filter_category = st.sidebar.radio("Filter Type:", ("Smoothing (Low Pass)", "Sharpening (High Pass)"))
        
        if filter_category == "Smoothing (Low Pass)":
            smooth_type = st.sidebar.selectbox("Method:", ["Averaging (Box)", "Gaussian", "Median"])
            k_size = st.sidebar.slider("Kernel Size (must be odd)", 3, 15, 3, step=2)
            
            if smooth_type == "Averaging (Box)":
                processed_image = cv2.blur(original_image, (k_size, k_size))
            elif smooth_type == "Gaussian":
                processed_image = cv2.GaussianBlur(original_image, (k_size, k_size), 0)
            elif smooth_type == "Median":
                processed_image = cv2.medianBlur(original_image, k_size)
                
        elif filter_category == "Sharpening (High Pass)":
            sharp_type = st.sidebar.selectbox("Method:", ["Laplacian", "Sobel", "Prewitt", "High-boost"])
            
            if sharp_type == "Laplacian":
                # Laplacian can produce negative values, so use CV_64F then abs
                lap = cv2.Laplacian(original_image, cv2.CV_64F)
                lap = np.uint8(np.absolute(lap))
                # Often added back to original to sharpen
                add_back = st.sidebar.checkbox("Add back to original (Sharpened Image)", value=True)
                if add_back:
                    # Simple sharpening: Original - Laplacian (depending on kernel center sign)
                    # Standard 3x3 Laplacian has -4 center, so we subtract. 
                    # Or if center is positive, we add. 
                    # Let's just show the Laplacian edges or the enhanced image.
                    processed_image = cv2.addWeighted(original_image, 1, lap, 1, 0) # Simple addition
                else:
                    processed_image = lap

            elif sharp_type == "Sobel":
                axis = st.sidebar.radio("Direction", ("X", "Y", "Combined"))
                k_size = st.sidebar.slider("Kernel Size", 3, 7, 3, step=2)
                
                sobelx = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=k_size)
                sobely = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=k_size)
                
                if axis == "X":
                    processed_image = np.uint8(np.absolute(sobelx))
                elif axis == "Y":
                    processed_image = np.uint8(np.absolute(sobely))
                else:
                    combined = cv2.magnitude(sobelx, sobely)
                    processed_image = np.uint8(cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX))

            elif sharp_type == "Prewitt":
                # Prewitt kernels
                kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
                kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
                
                img_prewittx = cv2.filter2D(original_image, -1, kernelx)
                img_prewitty = cv2.filter2D(original_image, -1, kernely)
                
                processed_image = img_prewittx + img_prewitty

            elif sharp_type == "High-boost":
                # Highboost = A * Original - LowPass
                # Highboost = (A-1) * Original + HighPass
                A = st.sidebar.slider("Amplification Factor (A)", 1.0, 5.0, 1.2)
                k_size = 3
                blurred = cv2.GaussianBlur(original_image, (k_size, k_size), 0)
                
                # Mask = Original - Blurred
                mask = original_image.astype(float) - blurred.astype(float)
                
                # Result = Original + k * Mask
                # If A is used as boost factor: Result = A * Original - Blurred ? 
                # Standard formula: g(x,y) = f(x,y) + k * g_mask(x,y)
                # where g_mask = f(x,y) - f_blur(x,y)
                # So g = f + k(f - f_blur) = (1+k)f - k*f_blur
                # Let's use the slider as 'k' (boost factor)
                
                k = A # Using A as the boost coefficient
                
                highboost = original_image.astype(float) + k * mask
                processed_image = np.clip(highboost, 0, 255).astype(np.uint8)


    # --- Display Result ---
    with col2:
        st.subheader("Processed Image")
        st.image(processed_image, use_container_width=True, clamp=True, channels='GRAY')
        
        if st.checkbox("Show Processed Histogram"):
            fig2, ax2 = plt.subplots()
            ax2.hist(processed_image.ravel(), 256, [0, 256])
            st.pyplot(fig2)

else:
    st.info("Please upload an image to get started.")
