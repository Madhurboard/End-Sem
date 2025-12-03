import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from PIL import Image

# Set page config
st.set_page_config(page_title="🎓 DIP Learning Tool", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

# Generate sample images programmatically (reliable, no network needed)
@st.cache_data
def generate_sample_images():
    """Generate sample images for testing"""
    samples = {}
    size = 256
    
    # 1. Lena-like (face-like pattern with smooth gradients)
    lena = np.zeros((size, size), dtype=np.uint8)
    # Create a face-like pattern
    cv2.ellipse(lena, (128, 128), (80, 100), 0, 0, 360, 180, -1)  # Face
    cv2.ellipse(lena, (100, 110), (15, 10), 0, 0, 360, 80, -1)   # Left eye
    cv2.ellipse(lena, (156, 110), (15, 10), 0, 0, 360, 80, -1)   # Right eye
    cv2.ellipse(lena, (128, 160), (20, 10), 0, 0, 180, 100, 2)   # Mouth
    # Add some texture
    noise = np.random.randint(0, 30, (size, size), dtype=np.uint8)
    lena = cv2.add(lena, noise)
    lena = cv2.GaussianBlur(lena, (5, 5), 2)
    samples["Portrait (Face Pattern)"] = lena
    
    # 2. Cameraman-like (figure with tripod)
    cam = np.ones((size, size), dtype=np.uint8) * 200  # Sky background
    cam[180:, :] = 100  # Ground
    # Person shape
    cv2.rectangle(cam, (100, 80), (150, 180), 60, -1)  # Body
    cv2.circle(cam, (125, 60), 25, 50, -1)  # Head
    # Tripod
    cv2.line(cam, (160, 120), (200, 200), 40, 3)
    cv2.line(cam, (160, 120), (180, 200), 40, 3)
    cv2.line(cam, (160, 120), (220, 200), 40, 3)
    cv2.rectangle(cam, (155, 100), (185, 130), 30, -1)  # Camera
    cam = cv2.GaussianBlur(cam, (3, 3), 1)
    samples["Cameraman (Figure)"] = cam
    
    # 3. Peppers-like (organic shapes with varying intensities)
    pep = np.ones((size, size), dtype=np.uint8) * 80
    # Create pepper-like shapes
    cv2.ellipse(pep, (80, 100), (50, 35), -20, 0, 360, 180, -1)
    cv2.ellipse(pep, (180, 90), (40, 50), 30, 0, 360, 120, -1)
    cv2.ellipse(pep, (130, 180), (45, 40), 10, 0, 360, 200, -1)
    cv2.ellipse(pep, (200, 180), (35, 30), -15, 0, 360, 150, -1)
    cv2.ellipse(pep, (60, 190), (30, 35), 5, 0, 360, 100, -1)
    # Add highlights
    for _ in range(20):
        x, y = np.random.randint(30, 226, 2)
        cv2.circle(pep, (x, y), np.random.randint(3, 8), np.random.randint(160, 255), -1)
    pep = cv2.GaussianBlur(pep, (7, 7), 3)
    samples["Peppers (Organic Shapes)"] = pep
    
    # 4. Moon (circular with craters)
    moon = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(moon, (128, 128), 100, 200, -1)  # Moon surface
    # Add craters
    craters = [(90, 100, 15), (150, 80, 20), (160, 150, 12), (100, 160, 18), (130, 120, 10)]
    for cx, cy, r in craters:
        cv2.circle(moon, (cx, cy), r, 140, -1)
        cv2.circle(moon, (cx-2, cy-2), r-3, 170, -1)  # Highlight
    moon = cv2.GaussianBlur(moon, (5, 5), 2)
    samples["Moon (Astronomy)"] = moon
    
    # 5. Building/Architecture
    bldg = np.ones((size, size), dtype=np.uint8) * 180  # Sky
    # Main building
    cv2.rectangle(bldg, (60, 80), (180, 240), 100, -1)
    # Windows grid
    for row in range(4):
        for col in range(4):
            x = 70 + col * 28
            y = 95 + row * 35
            cv2.rectangle(bldg, (x, y), (x+18, y+25), 50, -1)
    # Door
    cv2.rectangle(bldg, (100, 190), (140, 240), 60, -1)
    # Small building
    cv2.rectangle(bldg, (190, 150), (240, 240), 120, -1)
    samples["Building (Architecture)"] = bldg
    
    # 6. Gradient (for testing transforms)
    gradient = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    samples["Gradient (Horizontal)"] = gradient
    
    return samples

SAMPLE_IMAGES = generate_sample_images()

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
- ✂️ **Image Segmentation** - Dividing images into meaningful regions
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
    original_image = SAMPLE_IMAGES[sample_choice].copy()
        
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
            "4. Morphological Operations (Shape)",
            "5. Image Segmentation (Region)"
        )
    )

    processed_image = original_image.copy()
    
    # ==========================================
    # 1. INTENSITY / POINT TRANSFORMATIONS
    # ==========================================
    if "Intensity" in operation_type:
        st.header("1. Intensity Transformations")
        
        tab_vis, tab_theory = st.tabs(["🛠️ Interactive Lab", "📖 Theory & Formulas"])
        
        with tab_theory:
            st.markdown(r"""
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

    # ==========================================
    # 5. IMAGE SEGMENTATION
    # ==========================================
    elif "Segmentation" in operation_type:
        st.header("5. Image Segmentation")
        
        # Create tabs for the new structure
        seg_tabs = st.tabs([
            "1. Discontinuity (Edge)", 
            "2. Thresholding", 
            "3. Region-Based", 
            "4. Boundary & Corner",
            "📖 Theory"
        ])
        
        # --- Tab 1: Discontinuity (Edge-Based) ---
        with seg_tabs[0]:
            st.subheader("Detection of Discontinuities")
            disc_type = st.sidebar.selectbox(
                "Discontinuity Type:",
                ["Point Detection", "Line Detection", "Edge Detection (Gradient)", "Laplacian (2nd Derivative)"]
            )
            
            if disc_type == "Point Detection":
                # 2.3 Point Detection
                mask_type = st.sidebar.selectbox("Mask Type", ["Standard Laplacian", "Weighted"])
                thresh = st.sidebar.slider("Threshold", 0, 255, 127)
                
                if mask_type == "Standard Laplacian":
                    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                else:
                    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
                    
                filtered = cv2.filter2D(original_image, cv2.CV_64F, kernel)
                processed_image = np.uint8(np.abs(filtered) > thresh) * 255
                st.info("Detects isolated points using Laplacian masks.")
                
            elif disc_type == "Line Detection":
                # 2.4 Line Detection
                line_orient = st.sidebar.selectbox("Orientation", ["Horizontal", "Vertical", "+45 Diagonal", "-45 Diagonal"])
                thresh = st.sidebar.slider("Threshold", 0, 255, 127)
                
                masks = {
                    "Horizontal": np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]),
                    "Vertical": np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]),
                    "+45 Diagonal": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]),
                    "-45 Diagonal": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
                }
                
                filtered = cv2.filter2D(original_image, cv2.CV_64F, masks[line_orient])
                processed_image = np.uint8(np.abs(filtered) > thresh) * 255
                st.info(f"Detects lines oriented at {line_orient}.")
                
            elif disc_type == "Edge Detection (Gradient)":
                # 2.5 - 2.9 Gradient Operators
                operator = st.sidebar.selectbox("Operator", ["Roberts", "Prewitt", "Sobel"])
                
                if operator == "Roberts":
                    # 2.7 Roberts
                    kx = np.array([[1, 0], [0, -1]], dtype=np.float64)
                    ky = np.array([[0, 1], [-1, 0]], dtype=np.float64)
                    st.info("Roberts Cross Operator (2x2)")
                elif operator == "Prewitt":
                    # 2.8 Prewitt
                    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
                    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
                    st.info("Prewitt Operator (3x3)")
                else: # Sobel
                    # 2.9 Sobel
                    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
                    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
                    st.info("Sobel Operator (3x3) - Smoothing effect")
                
                Gx = cv2.filter2D(original_image.astype(np.float64), -1, kx)
                Gy = cv2.filter2D(original_image.astype(np.float64), -1, ky)
                magnitude = np.sqrt(Gx**2 + Gy**2)
                processed_image = np.clip(magnitude, 0, 255).astype(np.uint8)
                
            elif disc_type == "Laplacian (2nd Derivative)":
                # 2.10 Laplacian
                lap_type = st.sidebar.selectbox("Method", ["Standard Laplacian", "LoG (Laplacian of Gaussian)"])
                
                if lap_type == "Standard Laplacian":
                    processed_image = cv2.Laplacian(original_image, cv2.CV_8U)
                    st.info("Second-order derivative for edge detection.")
                else:
                    sigma = st.sidebar.slider("Sigma", 0.5, 5.0, 1.0)
                    blurred = cv2.GaussianBlur(original_image, (0,0), sigma)
                    processed_image = cv2.Laplacian(blurred, cv2.CV_8U)
                    st.info("LoG: Gaussian smoothing followed by Laplacian.")

        # --- Tab 2: Thresholding ---
        with seg_tabs[1]:
            st.subheader("Pixel-Based Segmentation (Thresholding)")
            thresh_mode = st.sidebar.selectbox("Thresholding Mode", 
                ["Single (Global)", "Multiple Thresholding", "Adaptive (Local)", "Optimal (Iterative)"])
            
            if thresh_mode == "Single (Global)":
                # 3.1 Single Global
                T = st.sidebar.slider("Threshold (T)", 0, 255, 127)
                _, processed_image = cv2.threshold(original_image, T, 255, cv2.THRESH_BINARY)
                st.info(f"Global Thresholding: T={T}")
                
            elif thresh_mode == "Multiple Thresholding":
                # 3.2 Multiple
                T1 = st.sidebar.slider("Threshold 1", 0, 255, 80)
                T2 = st.sidebar.slider("Threshold 2", 0, 255, 160)
                processed_image = np.zeros_like(original_image)
                processed_image[original_image > T1] = 127
                processed_image[original_image > T2] = 255
                st.info("Separates image into 3 classes (Black, Gray, White).")
                
            elif thresh_mode == "Adaptive (Local)":
                # 3.5 Adaptive
                method = st.sidebar.selectbox("Method", ["Mean", "Gaussian"])
                block_size = st.sidebar.slider("Block Size", 3, 99, 11, step=2)
                C = st.sidebar.slider("Constant C", -10, 10, 2)
                
                algo = cv2.ADAPTIVE_THRESH_MEAN_C if method == "Mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                processed_image = cv2.adaptiveThreshold(original_image, 255, algo, cv2.THRESH_BINARY, block_size, C)
                st.info("Local thresholding for uneven illumination.")
                
            elif thresh_mode == "Optimal (Iterative)":
                # 3.4 Iterative Selection
                T = 127.0 # Initial guess
                T_prev = 0
                iter_count = 0
                while abs(T - T_prev) > 0.5 and iter_count < 100:
                    T_prev = T
                    mu1 = original_image[original_image > T].mean() if len(original_image[original_image > T]) > 0 else 0
                    mu2 = original_image[original_image <= T].mean() if len(original_image[original_image <= T]) > 0 else 0
                    T = (mu1 + mu2) / 2
                    iter_count += 1
                
                _, processed_image = cv2.threshold(original_image, int(T), 255, cv2.THRESH_BINARY)
                st.success(f"Converged at T={int(T)} in {iter_count} iterations.")

        # --- Tab 3: Region-Based ---
        with seg_tabs[2]:
            st.subheader("Region-Based Segmentation")
            reg_method = st.sidebar.selectbox("Method", ["Region Growing", "Split-and-Merge"])
            
            if reg_method == "Region Growing":
                # 4.2 Region Growing
                seed_x = st.sidebar.slider("Seed X", 0, original_image.shape[1]-1, original_image.shape[1]//2)
                seed_y = st.sidebar.slider("Seed Y", 0, original_image.shape[0]-1, original_image.shape[0]//2)
                thresh = st.sidebar.slider("Similarity Threshold", 1, 50, 10)
                
                # Simple region growing
                h, w = original_image.shape
                mask = np.zeros((h+2, w+2), np.uint8)
                processed_image = original_image.copy()
                cv2.floodFill(processed_image, mask, (seed_x, seed_y), 255, loDiff=thresh, upDiff=thresh)
                # Highlight the region
                _, processed_image = cv2.threshold(mask[1:-1, 1:-1], 0, 255, cv2.THRESH_BINARY)
                st.info(f"Region growing from seed ({seed_x}, {seed_y}).")
                
            else:
                # 4.3 - 4.5 Split and Merge (Simulated with Quadtree decomposition visualization)
                st.info("Simulating Split-and-Merge (Quadtree Decomposition)")
                min_std = st.sidebar.slider("Min Std Dev", 0, 50, 10)
                
                # Recursive function for quadtree
                def split_image(img):
                    if img.std() <= min_std or img.shape[0] <= 4:
                        return np.full(img.shape, img.mean(), dtype=np.uint8)
                    
                    h, w = img.shape
                    h2, w2 = h//2, w//2
                    top_left = split_image(img[:h2, :w2])
                    top_right = split_image(img[:h2, w2:])
                    bot_left = split_image(img[h2:, :w2])
                    bot_right = split_image(img[h2:, w2:])
                    
                    # Reconstruct
                    top = np.hstack((top_left, top_right))
                    bot = np.hstack((bot_left, bot_right))
                    return np.vstack((top, bot))
                
                processed_image = split_image(original_image)

        # --- Tab 4: Boundary & Corner ---
        with seg_tabs[3]:
            st.subheader("Boundary & Corner Detection")
            feat_type = st.sidebar.selectbox("Feature", ["Corner Detection", "Boundary Extraction"])
            
            if feat_type == "Corner Detection":
                # 5.2 Corner Detection (Harris)
                block_size = st.sidebar.slider("Block Size", 2, 10, 2)
                k_size = st.sidebar.slider("Sobel Aperture", 1, 7, 3, step=2)
                k = st.sidebar.slider("Harris Parameter k", 0.01, 0.1, 0.04)
                
                dst = cv2.cornerHarris(original_image, block_size, k_size, k)
                dst = cv2.dilate(dst, None)
                
                # Draw corners on image
                processed_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
                processed_image[dst > 0.01 * dst.max()] = [0, 0, 255]
                st.info("Harris Corner Detection.")
                
            else:
                # 6.2 Boundary Detection
                kernel_size = st.sidebar.slider("Structuring Element Size", 3, 7, 3, step=2)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                erosion = cv2.erode(original_image, kernel, iterations=1)
                processed_image = original_image - erosion
                st.info("Boundary = Image - Erode(Image)")

        # --- Tab 5: Theory ---
        with seg_tabs[4]:
            st.markdown("""
            ### 1. Introduction to Segmentation
            **Goal:** Divide image into meaningful regions (objects).
            - **Discontinuity-based:** Detects edges, lines, points.
            - **Similarity-based:** Groups pixels with similar properties (thresholding, region growing).

            ### 2. Detection of Discontinuities (Edge-Based)
            **2.1 Types:** Point, Line, Edge.
            **2.2 Masks:** Convolution with 3x3 masks.
            **2.7 Roberts:** 2x2 cross-gradient operator.
            **2.8 Prewitt:** 3x3 masks (Horizontal/Vertical).
            **2.9 Sobel:** 3x3 masks with smoothing (1-2-1 weights).
            **2.10 Laplacian:** Second-order derivative, detects zero-crossings.

            ### 3. Thresholding (Pixel-Based)
            **3.1 Global:** $g(x,y) = 1$ if $f(x,y) > T$, else 0.
            **3.4 Iterative Selection:**
            1. Pick initial $T$.
            2. Split into groups $G_1, G_2$.
            3. Compute means $\mu_1, \mu_2$.
            4. Update $T = (\mu_1 + \mu_2) / 2$.
            5. Repeat until stable.
            **3.5 Adaptive:** Local thresholds for uneven illumination.

            ### 4. Region-Based Segmentation
            **4.2 Region Growing:** Start at seed, add neighbors if similar.
            **4.3 Region Splitting:** Split non-uniform regions (Quadtrees).
            **4.4 Region Merging:** Merge adjacent uniform regions.
            **4.5 Split-and-Merge:** Combine both for optimal results.

            ### 5. Line, Edge & Corner
            **5.2 Corner Detection:** Intensity variation in two directions.
            **5.3 Edge Linking:** Connect edge points to form boundaries.
            """)

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
        # Ensure images are the same size before computing difference
        target_shape = (original_image.shape[1], original_image.shape[0])
        if processed_image.shape[:2] != original_image.shape[:2]:
            processed_image_resized = cv2.resize(processed_image, target_shape)
        else:
            processed_image_resized = processed_image

        # Ensure both are grayscale for difference calculation
        if len(original_image.shape) == 3:
            orig_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = original_image
            
        if len(processed_image_resized.shape) == 3:
            proc_gray = cv2.cvtColor(processed_image_resized, cv2.COLOR_RGB2GRAY)
        else:
            proc_gray = processed_image_resized
            
        # Ensure same type
        orig_gray = orig_gray.astype(np.uint8)
        proc_gray = proc_gray.astype(np.uint8)

        diff = cv2.absdiff(orig_gray, proc_gray)
            
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
    
    feature_cols = st.columns(5)
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
    with feature_cols[4]:
        st.markdown("### 5️⃣ Segmentation")
        st.markdown("""
        - Global/Adaptive Threshold
        - Otsu's Method
        - Canny/Sobel Edge
        - Region Growing
        - Watershed
        """)
