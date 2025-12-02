import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from PIL import Image
import urllib.request

# Set page config
st.set_page_config(page_title="🎓 DIP Learning Tool", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

# Sample images from the web
SAMPLE_IMAGES = {
    "Lena (Standard)": "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
    "Cameraman": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Cameraman.tif/lossy-page1-256px-Cameraman.tif.jpg",
    "Peppers": "https://upload.wikimedia.org/wikipedia/en/8/8c/PSR_B1509-58_-_Chandra_X-ray_Observatory.jpg",
    "Moon (Astronomy)": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/FullMoon2010.jpg/280px-FullMoon2010.jpg",
}

@st.cache_data
def load_sample_image(url):
    """Load image from URL"""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            image_data = response.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize to reasonable size
            if max(img.shape) > 512:
                scale = 512 / max(img.shape)
                img = cv2.resize(img, None, fx=scale, fy=scale)
        return img
    except Exception as e:
        return None

def add_noise(image, noise_type, amount):
    """Add noise to image for testing filters"""
    noisy = image.copy().astype(np.float32)
    
    if noise_type == "Salt & Pepper":
        prob = amount / 100
        # Salt
        salt = np.random.random(image.shape) < prob / 2
        noisy[salt] = 255
        # Pepper
        pepper = np.random.random(image.shape) < prob / 2
        noisy[pepper] = 0
        
    elif noise_type == "Gaussian":
        sigma = amount
        gaussian = np.random.normal(0, sigma, image.shape)
        noisy = noisy + gaussian
        
    elif noise_type == "Speckle":
        speckle = np.random.randn(*image.shape) * amount
        noisy = noisy + noisy * speckle / 100
        
    elif noise_type == "Poisson":
        # Scale, add poisson, scale back
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(noisy * vals / 256) / vals * 256
    
    return np.clip(noisy, 0, 255).astype(np.uint8)

def calculate_metrics(original, processed):
    """Calculate image quality metrics"""
    # Ensure same shape
    if original.shape != processed.shape:
        return None
    
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)
    
    # MSE
    mse = np.mean((orig - proc) ** 2)
    
    # PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)
    
    # SSIM (simplified version)
    mu_x = np.mean(orig)
    mu_y = np.mean(proc)
    sigma_x = np.std(orig)
    sigma_y = np.std(proc)
    sigma_xy = np.mean((orig - mu_x) * (proc - mu_y))
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))
    
    return {'MSE': mse, 'PSNR': psnr, 'SSIM': ssim}

def get_image_download_link(img, filename="processed_image.png"):
    """Generate download link for image"""
    _, buffer = cv2.imencode('.png', img)
    b64 = base64.b64encode(buffer).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 Download Processed Image</a>'

st.title("🎓 Interactive Digital Image Processing Learning Tool")
st.markdown("""
Learn and experiment with **Digital Image Processing** techniques interactively!
- 📊 **Intensity Transformations** - Point operations on individual pixels
- 🔲 **Spatial Filtering** - Neighborhood operations using kernels
- 📈 **Frequency Domain** - Fourier transform based filtering  
- 🔷 **Morphological Operations** - Shape-based binary operations
""")

# --- Sidebar: Image Source Selection ---
st.sidebar.header("📷 1. Image Source")
image_source = st.sidebar.radio("Choose image source:", ["Upload Your Image", "Use Sample Image", "Generate Test Pattern"])

original_image = None

if image_source == "Upload Your Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tif"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
elif image_source == "Use Sample Image":
    sample_choice = st.sidebar.selectbox("Select sample image:", list(SAMPLE_IMAGES.keys()))
    with st.spinner(f"Loading {sample_choice}..."):
        original_image = load_sample_image(SAMPLE_IMAGES[sample_choice])
    if original_image is None:
        st.sidebar.warning("Could not load sample image. Please upload your own.")
        
elif image_source == "Generate Test Pattern":
    pattern_type = st.sidebar.selectbox("Pattern type:", ["Gradient", "Checkerboard", "Circles", "Text"])
    size = 256
    if pattern_type == "Gradient":
        original_image = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    elif pattern_type == "Checkerboard":
        original_image = np.zeros((size, size), dtype=np.uint8)
        original_image[::16, ::16] = 255
        original_image = cv2.resize(np.kron(np.indices((8, 8)).sum(axis=0) % 2, np.ones((32, 32))).astype(np.uint8) * 255, (size, size))
    elif pattern_type == "Circles":
        original_image = np.zeros((size, size), dtype=np.uint8)
        for r in range(20, 120, 20):
            cv2.circle(original_image, (size//2, size//2), r, 255, 2)
    elif pattern_type == "Text":
        original_image = np.zeros((size, size), dtype=np.uint8)
        cv2.putText(original_image, "DIP", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 5)
        cv2.putText(original_image, "Tool", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, 128, 3)

# --- Noise Addition Section ---
if original_image is not None:
    st.sidebar.markdown("---")
    st.sidebar.header("🔊 Add Noise (Optional)")
    add_noise_toggle = st.sidebar.checkbox("Add noise to test filters")
    
    if add_noise_toggle:
        noise_type = st.sidebar.selectbox("Noise Type:", ["Salt & Pepper", "Gaussian", "Speckle", "Poisson"])
        if noise_type == "Salt & Pepper":
            noise_amount = st.sidebar.slider("Noise Probability (%)", 1, 50, 10)
        elif noise_type == "Gaussian":
            noise_amount = st.sidebar.slider("Sigma (std dev)", 5, 100, 25)
        elif noise_type == "Speckle":
            noise_amount = st.sidebar.slider("Speckle Amount", 10, 100, 30)
        else:
            noise_amount = 1
        
        original_image = add_noise(original_image, noise_type, noise_amount)
        st.sidebar.success(f"✅ Added {noise_type} noise")

if original_image is not None:
    
    # Display Original Image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image (Grayscale)")
        st.image(original_image, use_container_width=True, clamp=True, channels='GRAY')
        
        # Image info
        st.caption(f"Size: {original_image.shape[1]}×{original_image.shape[0]} | Range: [{original_image.min()}-{original_image.max()}]")
        
        # Histogram of original
        if st.checkbox("Show Original Histogram"):
            fig, ax = plt.subplots()
            ax.hist(original_image.ravel(), 256, [0, 256])
            st.pyplot(fig)

    # --- Sidebar: Operation Selection ---
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ 2. Select Learning Module")
    operation_type = st.sidebar.radio(
        "Choose Module:",
        (
            "1. Intensity Transformations (Point)", 
            "2. Spatial Filtering (Neighborhood)",
            "3. Frequency Domain (Fourier)",
            "4. Morphological Operations (Shape)"
        )
    )

    processed_image = original_image.copy()
    
    # ==========================================
    # 1. INTENSITY / POINT TRANSFORMATIONS
    # ==========================================
    if "Intensity" in operation_type:
        st.header("1. Intensity Transformations")
        tab_vis, tab_theory = st.tabs(["🎨 Interactive Lab", "📖 Theory & Formulas"])
        
        with tab_theory:
            st.markdown(r"""
            ## 📚 Intensity Transformations (Point Operations)
            
            ### Concept
            These operations modify pixel values **individually** without considering neighbors.
            
            **General Formula:** $s = T(r)$
            - $r$ = Input pixel intensity (0-255)
            - $s$ = Output pixel intensity
            - $T$ = Transformation function
            
            ---
            
            ### 🔢 Available Transformations
            
            | Transformation | Formula | Use Case |
            |---------------|---------|----------|
            | **Negative** | $s = L-1-r$ | Enhance white details in dark regions |
            | **Log** | $s = c \cdot \log(1+r)$ | Expand dark values, compress bright |
            | **Power-law (Gamma)** | $s = c \cdot r^\gamma$ | Correct display gamma, enhance contrast |
            | **Contrast Stretch** | Piecewise linear | Expand limited range to full 0-255 |
            
            ---
            
            ### 💡 Tips
            - **γ < 1**: Brightens dark regions (screen correction)
            - **γ > 1**: Darkens image (printer correction)
            - **Log transform**: Great for images with large dynamic range (e.g., Fourier spectrum)
            """)
            
        with tab_vis:
            st.sidebar.subheader("Transformation Settings")
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
            
            # Initialize T(r) for plotting
            r_values = np.arange(256)
            s_values = r_values.copy()

            if transform_mode == "Image Negative":
                # s = L - 1 - r
                processed_image = 255 - original_image
                s_values = 255 - r_values
                st.sidebar.info("Formula: $s = 255 - r$")

            elif transform_mode == "Log Transformation":
                # s = c * log(1 + r)
                img_max = np.max(original_image)
                default_c = 255 / np.log(1 + float(img_max)) if img_max > 0 else 1
                max_c_slider = max(100.0, float(default_c) * 1.2)
                c = st.sidebar.slider("Constant (c)", 0.0, float(max_c_slider), float(default_c))
                
                processed_image = c * (np.log(1 + original_image.astype(np.float64)))
                processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
                
                s_values = c * np.log(1 + r_values)
                s_values = np.clip(s_values, 0, 255)
                st.sidebar.info(r"Formula: $s = c \cdot \log(1 + r)$")

            elif transform_mode == "Inverse Log Transformation":
                c = st.sidebar.slider("Constant (c)", 0.0, 5.0, 1.0)
                norm_img = original_image / 255.0
                processed_image = c * (np.exp(norm_img) - 1)
                processed_image = np.clip(processed_image * 255, 0, 255).astype(np.uint8)
                
                s_values = c * (np.exp(r_values/255.0) - 1) * 255
                s_values = np.clip(s_values, 0, 255)
                st.sidebar.info(r"Formula: $s = c \cdot (e^r - 1)$")

            elif transform_mode == "Power-law (Gamma) Transform":
                gamma = st.sidebar.slider("Gamma (γ)", 0.1, 5.0, 1.0, 0.1)
                c = st.sidebar.number_input("Constant (c)", value=1.0)
                
                norm_img = original_image / 255.0
                processed_image = c * np.power(norm_img, gamma)
                processed_image = np.clip(processed_image * 255, 0, 255).astype(np.uint8)
                
                s_values = c * np.power(r_values/255.0, gamma) * 255
                s_values = np.clip(s_values, 0, 255)
                
                st.sidebar.info(r"Formula: $s = c \cdot r^{\gamma}$")

            elif transform_mode == "Contrast Stretching":
                st.sidebar.text("Define points (r1, s1) and (r2, s2)")
                r1 = st.sidebar.slider("r1", 0, 255, 70)
                s1 = st.sidebar.slider("s1", 0, 255, 0)
                r2 = st.sidebar.slider("r2", 0, 255, 140)
                s2 = st.sidebar.slider("s2", 0, 255, 255)
                
                def pixel_val(pix, r1, s1, r2, s2):
                    if pix <= r1: return (s1 / r1) * pix if r1 > 0 else 0
                    elif pix <= r2: return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1 if r2 > r1 else s1
                    else: return ((255 - s2) / (255 - r2)) * (pix - r2) + s2 if 255 > r2 else s2
                
                pixel_val_vec = np.vectorize(pixel_val)
                processed_image = pixel_val_vec(original_image, r1, s1, r2, s2).astype(np.uint8)
                s_values = pixel_val_vec(r_values, r1, s1, r2, s2)

            elif transform_mode == "Gray-level Slicing":
                min_r = st.sidebar.slider("Min Intensity", 0, 255, 100)
                max_r = st.sidebar.slider("Max Intensity", 0, 255, 200)
                background = st.sidebar.radio("Background:", ("Retain Background", "Make Black"))
                
                if background == "Retain Background":
                    processed_image = original_image.copy()
                    mask = (original_image >= min_r) & (original_image <= max_r)
                    processed_image[mask] = 255
                    
                    s_values = r_values.copy()
                    mask_vals = (r_values >= min_r) & (r_values <= max_r)
                    s_values[mask_vals] = 255
                else:
                    processed_image = np.zeros_like(original_image)
                    mask = (original_image >= min_r) & (original_image <= max_r)
                    processed_image[mask] = 255
                    
                    s_values = np.zeros_like(r_values)
                    mask_vals = (r_values >= min_r) & (r_values <= max_r)
                    s_values[mask_vals] = 255

            elif transform_mode == "Bit-plane Slicing":
                plane = st.sidebar.slider("Bit Plane", 0, 7, 7)
                processed_image = cv2.bitwise_and(original_image, 2**plane)
                processed_image = processed_image * (255 // (2**plane))
                st.sidebar.text(f"Showing Bit Plane {plane}")
                # Plotting bit plane is tricky as it's not a continuous function, skip plot or show step
                s_values = np.zeros_like(r_values) # Placeholder

            # Plot Transformation Function
            if transform_mode != "Bit-plane Slicing":
                fig_t, ax_t = plt.subplots(figsize=(3, 3))
                ax_t.plot(r_values, s_values, 'r-')
                ax_t.set_title("Transformation T(r)")
                ax_t.set_xlabel("Input (r)")
                ax_t.set_ylabel("Output (s)")
                ax_t.grid(True)
                st.sidebar.pyplot(fig_t)

    # ==========================================
    # 2. NEIGHBORHOOD / SPATIAL FILTERING
    # ==========================================
    elif "Spatial Filtering" in operation_type:
        st.header("2. Spatial Filtering (Neighborhood Operations)")
        
        tab_vis2, tab_theory2 = st.tabs(["🔬 Interactive Lab", "📖 Theory & Kernels"])
        
        with tab_theory2:
            st.markdown(r"""
            ## 📚 Spatial Filtering
            
            ### Concept
            Unlike point operations, spatial filters use a **neighborhood** of pixels.
            A small matrix called a **kernel/mask** slides over the image.
            
            **Operation:** $g(x,y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} w(s,t) \cdot f(x+s, y+t)$
            
            ---
            
            ### 🔲 Common Kernels
            
            **Box (Averaging) Filter 3×3:**
            ```
            1/9  1/9  1/9
            1/9  1/9  1/9
            1/9  1/9  1/9
            ```
            
            **Gaussian 3×3:**
            ```
            1/16  2/16  1/16
            2/16  4/16  2/16
            1/16  2/16  1/16
            ```
            
            **Laplacian (Edge Detection):**
            ```
             0  -1   0
            -1   4  -1
             0  -1   0
            ```
            
            ---
            
            ### 📊 Filter Types
            
            | Type | Purpose | Example |
            |------|---------|---------|
            | **Smoothing (LPF)** | Blur, noise reduction | Box, Gaussian |
            | **Sharpening (HPF)** | Edge enhancement | Laplacian, Sobel |
            | **Order-Statistic** | Nonlinear filtering | Median (salt & pepper) |
            """)
        
        with tab_vis2:
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

    # ==========================================
    # 3. FREQUENCY DOMAIN FILTERING
    # ==========================================
    elif "Frequency Domain" in operation_type:
        st.header("3. Frequency Domain Filtering")
        
        tab_vis, tab_theory = st.tabs(["🎛️ Interactive Lab", "📘 Theory & Formulas"])
        
        with tab_theory:
            st.markdown(r"""
            ### Concept
            Filtering in the frequency domain involves:
            1.  **DFT**: Transform image to frequency domain ($F(u,v)$).
            2.  **Shift**: Move zero-frequency component to center.
            3.  **Filter**: Multiply by a mask $H(u,v)$.
                $$G(u,v) = H(u,v) \times F(u,v)$$
            4.  **IDFT**: Inverse transform to get back spatial image.
            
            ### Filters
            - **Low Pass**: Blurs image (removes high freq edges).
            - **High Pass**: Sharpens image (removes low freq smooth areas).
            """)

        with tab_vis:
            st.sidebar.subheader("Frequency Domain Settings")
            
            # Helper function to create distance matrix
            def get_distance_matrix(shape):
                rows, cols = shape
                crow, ccol = rows // 2, cols // 2
                x = np.linspace(-ccol, ccol - 1, cols)
                y = np.linspace(-crow, crow - 1, rows)
                X, Y = np.meshgrid(x, y)
                dist = np.sqrt(X**2 + Y**2)
                return dist

            freq_filter_type = st.sidebar.selectbox(
                "Filter Type:",
                ["Ideal Low-Pass (ILPF)", "Butterworth Low-Pass (BLPF)", "Gaussian Low-Pass (GLPF)",
                 "Ideal High-Pass (IHPF)", "Butterworth High-Pass (BHPF)", "Gaussian High-Pass (GHPF)",
                 "Band-Pass Filter", "Homomorphic Filtering"]
            )

            # Perform DFT
            dft = cv2.dft(np.float32(original_image), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            
            # Calculate Magnitude Spectrum of Original
            mag_original = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
            
            rows, cols = original_image.shape
            crow, ccol = rows // 2, cols // 2
            
            # Initialize mask
            mask = np.zeros((rows, cols, 2), np.float32)
            D = get_distance_matrix((rows, cols))
            
            cutoff = st.sidebar.slider("Cutoff Frequency (Radius D0)", 1, min(rows, cols)//2, 30)
            
            order = 2
            if "Butterworth" in freq_filter_type:
                order = st.sidebar.slider("Order (n)", 1, 10, 2)
                
            if freq_filter_type == "Ideal Low-Pass (ILPF)":
                mask[D <= cutoff] = 1
                st.sidebar.info("Passes frequencies within D0, cuts off others. Causes ringing.")
                
            elif freq_filter_type == "Butterworth Low-Pass (BLPF)":
                # H = 1 / (1 + (D/D0)^(2n))
                H = 1 / (1 + (D / cutoff)**(2 * order))
                mask[:, :, 0] = H
                mask[:, :, 1] = H
                st.sidebar.info("Smoother transition than Ideal. No ringing.")
                
            elif freq_filter_type == "Gaussian Low-Pass (GLPF)":
                # H = exp(-D^2 / (2*D0^2))
                H = np.exp(-(D**2) / (2 * (cutoff**2)))
                mask[:, :, 0] = H
                mask[:, :, 1] = H
                st.sidebar.info("Smooth transition. No ringing.")
                
            elif freq_filter_type == "Ideal High-Pass (IHPF)":
                mask[D > cutoff] = 1
                st.sidebar.info("Cuts off low frequencies. Sharpens but causes ringing.")
                
            elif freq_filter_type == "Butterworth High-Pass (BHPF)":
                # H = 1 / (1 + (D0/D)^(2n))
                with np.errstate(divide='ignore', invalid='ignore'):
                    H = 1 / (1 + (cutoff / D)**(2 * order))
                    H[D == 0] = 0 # Handle center
                mask[:, :, 0] = H
                mask[:, :, 1] = H
                st.sidebar.info("Smoother transition high-pass.")
                
            elif freq_filter_type == "Gaussian High-Pass (GHPF)":
                # H = 1 - exp(-D^2 / (2*D0^2))
                H = 1 - np.exp(-(D**2) / (2 * (cutoff**2)))
                mask[:, :, 0] = H
                mask[:, :, 1] = H
                st.sidebar.info("Smooth high-pass filter.")
                
            elif freq_filter_type == "Band-Pass Filter":
                bandwidth = st.sidebar.slider("Bandwidth (W)", 5, 100, 20)
                center_freq = st.sidebar.slider("Center Frequency (C0)", cutoff, min(rows, cols)//2, cutoff + 20)
                
                # Simple Ideal Band Pass: 1 if C0 - W/2 <= D <= C0 + W/2
                lower = center_freq - bandwidth / 2
                upper = center_freq + bandwidth / 2
                mask[(D >= lower) & (D <= upper)] = 1
                st.sidebar.info("Passes a band of frequencies.")

            elif freq_filter_type == "Homomorphic Filtering":
                # Homomorphic: ln(f) -> DFT -> H(u,v) -> IDFT -> exp(g)
                # H(u,v) = (gH - gL) * (1 - exp(-c * (D^2 / D0^2))) + gL
                gH = st.sidebar.slider("High Freq Gain (gH > 1)", 1.0, 5.0, 2.0)
                gL = st.sidebar.slider("Low Freq Gain (gL < 1)", 0.0, 1.0, 0.5)
                c_const = st.sidebar.slider("Constant (c)", 0.1, 5.0, 1.0)
                
                # Log transform
                img_log = np.log1p(np.float32(original_image))
                
                # DFT of log
                dft_log = cv2.dft(img_log, flags=cv2.DFT_COMPLEX_OUTPUT)
                dft_shift_log = np.fft.fftshift(dft_log)
                
                # Filter
                H = (gH - gL) * (1 - np.exp(-c_const * (D**2) / (cutoff**2))) + gL
                mask[:, :, 0] = H
                mask[:, :, 1] = H
                
                # Apply filter
                fshift = dft_shift_log * mask
                f_ishift = np.fft.ifftshift(fshift)
                img_back_log = cv2.idft(f_ishift)
                img_back_log = cv2.magnitude(img_back_log[:,:,0], img_back_log[:,:,1])
                
                # Exp transform
                processed_image = np.expm1(img_back_log)
                processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                st.sidebar.info("Enhances contrast by compressing dynamic range (illumination) and enhancing contrast (reflectance).")

            # Apply mask and Inverse DFT (for non-Homomorphic)
            if freq_filter_type != "Homomorphic Filtering":
                fshift = dft_shift * mask
                
                # Calculate Magnitude Spectrum of Filtered
                mag_filtered = 20 * np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1]) + 1)
                
                f_ishift = np.fft.ifftshift(fshift)
                img_back = cv2.idft(f_ishift)
                img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
                
                processed_image = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                # For homomorphic, we already computed processed_image, but let's just show original spectrum for filtered slot or skip
                mag_filtered = np.zeros_like(mag_original)

            # Visualization
            st.write("### 🔍 Frequency Domain Analysis")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.image(original_image, caption="1. Original Spatial Image", use_container_width=True, channels='GRAY')
                st.image(cv2.normalize(mag_original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), 
                         caption="2. Original Frequency Spectrum (DFT)", use_container_width=True, clamp=True)
                
            with c2:
                # Visualize Mask (Take channel 0)
                mask_vis = mask[:,:,0]
                st.image(mask_vis, caption=f"3. Filter Mask H(u,v) (Radius={cutoff})", use_container_width=True, clamp=True)
                st.markdown(f"**Filter:** {freq_filter_type}")
                
            with c3:
                st.image(processed_image, caption="5. Result Spatial Image (IDFT)", use_container_width=True, channels='GRAY')
                if freq_filter_type != "Homomorphic Filtering":
                    st.image(cv2.normalize(mag_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), 
                             caption="4. Filtered Spectrum (Spectrum × Mask)", use_container_width=True, clamp=True)

    # ==========================================
    # 4. MORPHOLOGICAL OPERATIONS
    # ==========================================
    elif "Morphological" in operation_type:
        st.header("4. Morphological Operations")
        
        tab_vis4, tab_theory4 = st.tabs(["🔬 Interactive Lab", "📖 Theory"])
        
        with tab_theory4:
            st.markdown(r"""
            ## 📚 Morphological Operations
            
            ### Concept
            Morphological operations work on **binary images** using a **structuring element (SE)**.
            They analyze and modify geometric structures in images.
            
            ---
            
            ### 🔷 Basic Operations
            
            | Operation | Symbol | Effect |
            |-----------|--------|--------|
            | **Dilation** | $A \oplus B$ | Expands/grows white regions |
            | **Erosion** | $A \ominus B$ | Shrinks white regions |
            
            ---
            
            ### 🔶 Combined Operations
            
            | Operation | Formula | Use Case |
            |-----------|---------|----------|
            | **Opening** | $(A \ominus B) \oplus B$ | Remove small bright spots (noise) |
            | **Closing** | $(A \oplus B) \ominus B$ | Fill small dark holes |
            
            ---
            
            ### 📐 Structuring Element
            The SE is a small shape (rectangle, cross, ellipse) that probes the image.
            - **Origin**: Usually the center pixel
            - **Size**: Controls the scale of the operation
            
            ### 💡 Applications
            - Noise removal
            - Object counting (connected components)
            - Boundary extraction
            - Skeleton extraction
            - Document image cleanup
            """)
        
        with tab_vis4:
            st.sidebar.subheader("Morphological Operations")
            
            # Thresholding for binary operations
            st.sidebar.markdown("### Pre-processing")
            is_binary = st.sidebar.checkbox("Convert to Binary first?", value=True)
            if is_binary:
                thresh_val = st.sidebar.slider("Threshold Value", 0, 255, 127)
                _, working_image = cv2.threshold(original_image, thresh_val, 255, cv2.THRESH_BINARY)
                st.sidebar.image(working_image, caption="Binary Input", width=150)
            else:
                working_image = original_image
                
            morph_cat = st.sidebar.selectbox(
                "Category:",
                ["Basic Morphology", "Combined Operations", "Advanced Transformations", "Boundary Extraction"]
            )
            
            se_shape = st.sidebar.selectbox("SE Shape:", ["Rectangle", "Ellipse", "Cross"])
            k_size = st.sidebar.slider("Structuring Element Size", 3, 21, 3, step=2)
            
            if se_shape == "Rectangle":
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            elif se_shape == "Ellipse":
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_size, k_size))
            
            # Show SE
            st.sidebar.markdown("**Structuring Element:**")
            se_display = (kernel * 255).astype(np.uint8)
            st.sidebar.image(cv2.resize(se_display, (60, 60), interpolation=cv2.INTER_NEAREST), width=60)
            
            if morph_cat == "Basic Morphology":
                op = st.sidebar.radio("Operation:", ["Dilation", "Erosion"])
                if op == "Dilation":
                    processed_image = cv2.dilate(working_image, kernel, iterations=1)
                    st.sidebar.info("Expands white regions. Fills holes.")
                else:
                    processed_image = cv2.erode(working_image, kernel, iterations=1)
                    st.sidebar.info("Shrinks white regions. Removes small objects.")
                    
            elif morph_cat == "Combined Operations":
                op = st.sidebar.radio("Operation:", ["Opening", "Closing"])
                if op == "Opening":
                    # Erosion followed by Dilation
                    processed_image = cv2.morphologyEx(working_image, cv2.MORPH_OPEN, kernel)
                    st.sidebar.info("Removes small objects (noise) from foreground.")
                else:
                    # Dilation followed by Erosion
                    processed_image = cv2.morphologyEx(working_image, cv2.MORPH_CLOSE, kernel)
                    st.sidebar.info("Fills small holes in foreground objects.")
                    
            elif morph_cat == "Advanced Transformations":
                adv_op = st.sidebar.selectbox("Operation:", 
                    ["Hit-or-Miss", "Morphological Gradient", "Top Hat", "Black Hat", "Skeletonization"])
                
                if adv_op == "Hit-or-Miss":
                    st.sidebar.warning("Hit-or-Miss requires a specific kernel to find a pattern. Using a simple cross kernel for demo.")
                    # Define a simple kernel for hit-or-miss (finding corners/isolated pixels etc)
                    hm_kernel = np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]], dtype=np.int8)
                    processed_image = cv2.morphologyEx(working_image, cv2.MORPH_HITMISS, hm_kernel)
                    
                elif adv_op == "Morphological Gradient":
                    processed_image = cv2.morphologyEx(working_image, cv2.MORPH_GRADIENT, kernel)
                    st.sidebar.info("Difference between Dilation and Erosion. Outlines edges.")
                    
                elif adv_op == "Top Hat":
                    processed_image = cv2.morphologyEx(working_image, cv2.MORPH_TOPHAT, kernel)
                    st.sidebar.info("Input image minus Opening. Highlights bright details.")
                    
                elif adv_op == "Black Hat":
                    processed_image = cv2.morphologyEx(working_image, cv2.MORPH_BLACKHAT, kernel)
                    st.sidebar.info("Closing minus Input image. Highlights dark details.")
                    
                elif adv_op == "Skeletonization":
                    # Implement skeletonization using OpenCV (Zhang-Suen thinning)
                    # Since skimage may not be available, use iterative erosion approach
                    skel = np.zeros(working_image.shape, np.uint8)
                    temp = working_image.copy()
                    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
                    
                    while True:
                        eroded = cv2.erode(temp, element)
                        dilated = cv2.dilate(eroded, element)
                        diff = cv2.subtract(temp, dilated)
                        skel = cv2.bitwise_or(skel, diff)
                        temp = eroded.copy()
                        
                        if cv2.countNonZero(temp) == 0:
                            break
                    
                    processed_image = skel
                    st.sidebar.info("Reduces foreground regions to 1-pixel wide skeleton.")

            elif morph_cat == "Boundary Extraction":
                # Boundary = A - Erosion(A)
                eroded = cv2.erode(working_image, kernel, iterations=1)
                processed_image = cv2.subtract(working_image, eroded)
                st.sidebar.info("Subtracting eroded image from original gives the boundary.")

    # --- Display Result ---
    with col2:
        st.subheader("Processed Image")
        st.image(processed_image, use_container_width=True, clamp=True, channels='GRAY')
        
        # Download button
        st.markdown(get_image_download_link(processed_image), unsafe_allow_html=True)
        
        if st.checkbox("Show Processed Histogram"):
            fig2, ax2 = plt.subplots()
            ax2.hist(processed_image.ravel(), 256, [0, 256])
            st.pyplot(fig2)
    
    # --- Metrics and Comparison Section ---
    st.markdown("---")
    st.header("📊 Analysis & Comparison")
    
    analysis_tabs = st.tabs(["📈 Image Metrics", "🔍 Difference View", "📊 Histogram Comparison"])
    
    with analysis_tabs[0]:
        metrics = calculate_metrics(original_image, processed_image)
        if metrics:
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("MSE", f"{metrics['MSE']:.2f}", help="Mean Squared Error - Lower is more similar")
            with mcol2:
                psnr_val = f"{metrics['PSNR']:.2f} dB" if metrics['PSNR'] != float('inf') else "∞ (Identical)"
                st.metric("PSNR", psnr_val, help="Peak Signal-to-Noise Ratio - Higher is better")
            with mcol3:
                st.metric("SSIM", f"{metrics['SSIM']:.4f}", help="Structural Similarity - 1.0 means identical")
    
    with analysis_tabs[1]:
        # Difference image
        diff = cv2.absdiff(original_image, processed_image)
        diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
        
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.image(diff, caption="Absolute Difference (Grayscale)", use_container_width=True, clamp=True)
        with dcol2:
            st.image(diff_colored, caption="Difference Heatmap (Blue=Same, Red=Different)", use_container_width=True)
    
    with analysis_tabs[2]:
        fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.hist(original_image.ravel(), 256, range=[0, 256], color='blue', alpha=0.7)
        ax1.set_title("Original Histogram")
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Frequency")
        
        ax2.hist(processed_image.ravel(), 256, range=[0, 256], color='green', alpha=0.7)
        ax2.set_title("Processed Histogram")
        ax2.set_xlabel("Intensity")
        ax2.set_ylabel("Frequency")
        
        plt.tight_layout()
        st.pyplot(fig_comp)
        
        # Overlay histogram
        fig_overlay, ax_overlay = plt.subplots(figsize=(10, 4))
        ax_overlay.hist(original_image.ravel(), 256, range=[0, 256], color='blue', alpha=0.5, label='Original')
        ax_overlay.hist(processed_image.ravel(), 256, range=[0, 256], color='green', alpha=0.5, label='Processed')
        ax_overlay.legend()
        ax_overlay.set_title("Overlaid Histograms")
        ax_overlay.set_xlabel("Intensity")
        ax_overlay.set_ylabel("Frequency")
        st.pyplot(fig_overlay)

else:
    st.info("👆 Please select an image source from the sidebar to get started.")
    
    # Show feature overview when no image loaded
    st.markdown("---")
    st.header("🎯 What You Can Learn")
    
    feature_cols = st.columns(4)
    with feature_cols[0]:
        st.markdown("### 1️⃣ Intensity")
        st.markdown("""
        - Image Negative
        - Log Transform
        - Gamma Correction
        - Contrast Stretching
        - Bit-plane Slicing
        """)
    with feature_cols[1]:
        st.markdown("### 2️⃣ Spatial")
        st.markdown("""
        - Box Filter
        - Gaussian Blur
        - Median Filter
        - Sobel Edge
        - Laplacian
        """)
    with feature_cols[2]:
        st.markdown("### 3️⃣ Frequency")
        st.markdown("""
        - Ideal LPF/HPF
        - Butterworth
        - Gaussian Filter
        - Band-pass
        - Homomorphic
        """)
    with feature_cols[3]:
        st.markdown("### 4️⃣ Morphology")
        st.markdown("""
        - Dilation/Erosion
        - Opening/Closing
        - Skeletonization
        - Boundary Extract
        - Hit-or-Miss
        """)
