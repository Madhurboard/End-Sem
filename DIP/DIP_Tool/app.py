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
        filter_category = st.sidebar.radio(
            "Filter Category:", 
            ("Smoothing Spatial Filters (Linear)", 
             "Ordered-Statistic Filters (Nonlinear)", 
             "Sharpening Spatial Filters")
        )
        
        # --- 1. Smoothing Spatial Filters (Linear) ---
        if filter_category == "Smoothing Spatial Filters (Linear)":
            smooth_type = st.sidebar.selectbox(
                "Method:", 
                ["Box (Averaging) Filter", "Weighted Averaging Filter", "Gaussian Smoothing Filter"]
            )
            k_size = st.sidebar.slider("Kernel Size (must be odd)", 3, 15, 3, step=2)
            
            if smooth_type == "Box (Averaging) Filter":
                processed_image = cv2.blur(original_image, (k_size, k_size))
                st.sidebar.info("Uniform weights. Smooths image but blurs edges.")
                
            elif smooth_type == "Weighted Averaging Filter":
                # Approximation of Gaussian using integer weights
                if k_size == 3:
                    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
                    processed_image = cv2.filter2D(original_image, -1, kernel)
                    st.sidebar.info("3x3 Weighted Average Kernel applied.")
                else:
                    st.sidebar.warning("Specific Weighted Average matrix is typically defined for 3x3. Using GaussianBlur for larger sizes.")
                    processed_image = cv2.GaussianBlur(original_image, (k_size, k_size), 0)

            elif smooth_type == "Gaussian Smoothing Filter":
                sigma = st.sidebar.slider("Sigma (σ)", 0.1, 10.0, 1.0)
                processed_image = cv2.GaussianBlur(original_image, (k_size, k_size), sigma)
                st.sidebar.info("True Gaussian distribution. Best for noise reduction while preserving edges better than Box.")

        # --- 2. Ordered-Statistic Filters (Nonlinear) ---
        elif filter_category == "Ordered-Statistic Filters (Nonlinear)":
            stat_type = st.sidebar.selectbox(
                "Method:", 
                ["Median Filter", "Max Filter", "Min Filter", "Midpoint Filter"]
            )
            k_size = st.sidebar.slider("Kernel Size (must be odd)", 3, 15, 3, step=2)
            
            if stat_type == "Median Filter":
                processed_image = cv2.medianBlur(original_image, k_size)
                st.sidebar.info("Replaces pixel with median of neighbors. Best for Salt & Pepper noise.")
                
            elif stat_type == "Max Filter":
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                processed_image = cv2.dilate(original_image, kernel)
                st.sidebar.info("Replaces pixel with max of neighbors. Removes pepper noise (brightens).")
                
            elif stat_type == "Min Filter":
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                processed_image = cv2.erode(original_image, kernel)
                st.sidebar.info("Replaces pixel with min of neighbors. Removes salt noise (darkens).")
                
            elif stat_type == "Midpoint Filter":
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                max_img = cv2.dilate(original_image, kernel)
                min_img = cv2.erode(original_image, kernel)
                # Midpoint = (Max + Min) / 2
                processed_image = ((max_img.astype(np.float32) + min_img.astype(np.float32)) / 2.0).astype(np.uint8)
                st.sidebar.info("Average of Max and Min in the window. Good for random noise (Gaussian/Uniform).")

        # --- 3. Sharpening Spatial Filters ---
        elif filter_category == "Sharpening Spatial Filters":
            sharp_cat = st.sidebar.radio("Sharpening Type:", 
                ("First-Order Derivative (Gradient)", "Second-Order Derivative (Laplacian)", "General Sharpening"))
            
            if sharp_cat == "First-Order Derivative (Gradient)":
                grad_type = st.sidebar.selectbox("Operator:", ["Sobel", "Prewitt"])
                
                if grad_type == "Sobel":
                    k_size = st.sidebar.slider("Kernel Size", 3, 7, 3, step=2)
                    axis = st.sidebar.radio("Direction", ("X", "Y", "Combined"))
                    
                    sobelx = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=k_size)
                    sobely = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=k_size)
                    
                    if axis == "X":
                        processed_image = np.uint8(np.absolute(sobelx))
                    elif axis == "Y":
                        processed_image = np.uint8(np.absolute(sobely))
                    else:
                        combined = cv2.magnitude(sobelx, sobely)
                        processed_image = np.uint8(cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX))
                    st.sidebar.info("Sobel: Smooths slightly then computes derivative. Good noise immunity.")

                elif grad_type == "Prewitt":
                    # Prewitt kernels
                    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
                    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
                    
                    img_prewittx = cv2.filter2D(original_image, -1, kernelx)
                    img_prewitty = cv2.filter2D(original_image, -1, kernely)
                    
                    # Combine magnitude
                    processed_image = cv2.addWeighted(np.abs(img_prewittx), 0.5, np.abs(img_prewitty), 0.5, 0)
                    st.sidebar.info("Prewitt: Simple derivative masks. Detects edges.")

            elif sharp_cat == "Second-Order Derivative (Laplacian)":
                k_size = st.sidebar.slider("Kernel Size", 1, 7, 3, step=2)
                # Laplacian
                lap = cv2.Laplacian(original_image, cv2.CV_64F, ksize=k_size)
                lap = np.uint8(np.absolute(lap))
                
                mode = st.sidebar.radio("Output Mode:", ("Laplacian Edges Only", "Sharpened Image (Original + Laplacian)"))
                
                if mode == "Laplacian Edges Only":
                    processed_image = lap
                else:
                    processed_image = cv2.addWeighted(original_image, 1, lap, 1, 0)
                st.sidebar.info("Laplacian: Isotropic (rotation invariant). Detects discontinuities.")

            elif sharp_cat == "General Sharpening":
                gen_type = st.sidebar.selectbox("Method:", ["Unsharp Masking", "High-boost Filtering"])
                
                k_size = st.sidebar.slider("Blur Kernel Size", 3, 15, 3, step=2)
                sigma = st.sidebar.slider("Blur Sigma", 1.0, 10.0, 1.0)
                
                blurred = cv2.GaussianBlur(original_image, (k_size, k_size), sigma)
                mask = original_image.astype(float) - blurred.astype(float)
                
                if gen_type == "Unsharp Masking":
                    k = 1.0
                    st.sidebar.info("Formula: Sharpened = Original + (Original - Blurred)")
                else:
                    # High-boost
                    A = st.sidebar.slider("Amplification Factor (A)", 1.1, 5.0, 1.2)
                    # High-boost = A * Original - Blurred
                    #            = (A-1)*Original + Original - Blurred
                    #            = (A-1)*Original + Mask
                    # This is one interpretation. Another is k * Mask.
                    # Let's use the formula: High-boost = A * Original - Blurred
                    k = 1.0 # Placeholder, we calculate directly below
                    
                if gen_type == "High-boost Filtering":
                    # Formula: A * Original - Blurred
                    sharpened = A * original_image.astype(float) - blurred.astype(float)
                else:
                    # Unsharp Masking: Original + Mask
                    sharpened = original_image.astype(float) + mask

                processed_image = np.clip(sharpened, 0, 255).astype(np.uint8)


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
