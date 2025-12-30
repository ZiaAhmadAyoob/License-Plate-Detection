import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="ANPR Traffic System",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Traffic/Security Theme) ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #004aad;
        color: white;
    }
    .stButton>button:hover {
        background-color: #00337a;
        color: white;
    }
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #004aad;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
col1, col2 = st.columns([1, 8])
with col1:
    st.markdown("# 🚔")
with col2:
    st.title("Smart Traffic Enforcement System")
    st.markdown("**Automatic Number Plate Recognition (ANPR)** | Powered by YOLOv11")

st.markdown("---")

# --- Sidebar Settings ---
st.sidebar.title("⚙️ System Config")
st.sidebar.subheader("Detection Settings")

# Confidence Slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.45, 
    step=0.01,
    help="Higher confidence reduces false positives (e.g., mistaking signs for plates)."
)

# App Mode Selector
app_mode = st.sidebar.selectbox(
    "Select Operation Mode",
    ["Parking Audit (Image)", "Traffic Surveillance (Video)", "Checkpoint (Live)"]
)

st.sidebar.markdown("---")
st.sidebar.info("Ensure your trained 'best.pt' (License Plate Model) is in the root directory.")

# --- Model Loading (Absolute Path Fix) ---
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Get absolute path to ensure file is found
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best.pt")

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error(f"⚠️ Model file not found at: {model_path}")
    st.info("Please rename your trained license plate model to 'best.pt' and place it here.")
    model = None

# --- Main Logic ---

if model:
    # ---------------- IMAGE MODE (Parking Audit) ----------------
    if app_mode == "Parking Audit (Image)":
        st.subheader("📸 Vehicle Entry Audit")
        uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Entry Vehicle", use_container_width=True)

            with col2:
                if st.button("Scan License Plate", type="primary"):
                    with st.spinner("Processing..."):
                        results = model(img_array, conf=conf_threshold)
                        annotated_img = results[0].plot()
                        
                        st.image(annotated_img, caption="Detection Result", use_container_width=True)
                        
                        # Vehicle Counter
                        count = len(results[0].boxes)
                        st.metric(label="Plates Detected", value=count)
                        
                        if count > 0:
                            st.success("✅ License plate identified.")
                        else:
                            st.warning("⚠️ No license plate found. Check image clarity.")

    # ---------------- VIDEO MODE (Traffic Surveillance) ----------------
    elif app_mode == "Traffic Surveillance (Video)":
        st.subheader("🎥 Traffic Flow Analysis")
        video_file = st.file_uploader("Upload CCTV footage", type=["mp4", "avi", "mov"])

        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            st.sidebar.markdown("---")
            stop_button = st.sidebar.button("Stop Surveillance")

            col1, col2 = st.columns([3, 1])
            with col1:
                stframe = st.empty()
            with col2:
                st.markdown("### 📊 Live Stats")
                kpi_text = st.empty()

            cap = cv2.VideoCapture(tfile.name)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop_button:
                    break

                results = model(frame, conf=conf_threshold)
                annotated = results[0].plot()
                
                # Display Frame
                stframe.image(annotated, channels="BGR", use_container_width=True)
                
                # Update Counter
                plate_count = len(results[0].boxes)
                kpi_text.markdown(f"**Plates in View:** {plate_count}")

            cap.release()
            st.success("Surveillance sequence ended.")

    # ---------------- LIVE MODE (Checkpoint) ----------------
    elif app_mode == "Checkpoint (Live)":
        st.subheader("🔴 Live Security Checkpoint")
        st.write("Active monitoring for incoming vehicles.")

        run = st.checkbox('Activate Camera Feed', value=False)
        
        frame_window = st.image([])
        cap = cv2.VideoCapture(0)

        if run:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera connection failed.")
                    break
                
                results = model(frame, conf=conf_threshold)
                annotated_frame = results[0].plot()
                
                # Color Correction for Streamlit
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                frame_window.image(annotated_frame)
        else:
            cap.release()
            st.write("Checkpoint camera is offline.")