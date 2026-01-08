import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import plotly.graph_objects as go
import time

# --------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Vehicle Classification AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to inject a "Pro" look (Dark theme friendly)
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #0E1117; 
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #FAFAFA;
    }
    /* Customizing the file uploader */
    .stFileUploader {
        border: 2px dashed #4B4B4B;
        border-radius: 10px;
        padding: 20px;
    }
    /* Metric container styling */
    div[data-testid="metric-container"] {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 2. CONSTANTS & MODEL LOADING
# --------------------------------------------------------------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = 'vehicle_efficientnet_best.keras'

# Based on your notebook's folder structure (Alphabetical Order)
CLASS_NAMES = [
    'Auto Rickshaw', 
    'Bike', 
    'Car', 
    'Motorcycle', 
    'Plane', 
    'Ship', 
    'Train'
]

@st.cache_resource
def load_trained_model():
    """
    Loads the model once and caches it to avoid reloading on every interaction.
    """
    try:
        # Load model with custom objects if necessary, though EfficientNet is standard
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --------------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------------------------------------
def preprocess_image(image):
    """
    Preprocesses the image to match EfficientNet requirements:
    1. Resize to (224, 224)
    2. Convert to Array
    3. Expand dimensions to (1, 224, 224, 3)
    4. Apply EfficientNet preprocessing
    """
    # Resize image
    image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    
    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    
    # Expand dims (1, 224, 224, 3)
    img_array = tf.expand_dims(img_array, 0)
    
    # Apply specific EfficientNet preprocessing (as done in your notebook)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    return img_array

def plot_confidence_chart(predictions):
    """
    Creates a cool interactive bar chart using Plotly.
    """
    probs = predictions[0]
    
    # Create colors: Highlight the max probability
    colors = ['#262730'] * len(CLASS_NAMES)
    max_idx = np.argmax(probs)
    colors[max_idx] = '#FF4B4B'  # Streamlit Red for the winner

    fig = go.Figure(data=[go.Bar(
        x=CLASS_NAMES,
        y=probs,
        text=[f"{p:.1%}" for p in probs],
        textposition='auto',
        marker_color=colors
    )])

    fig.update_layout(
        title="Confidence Distribution",
        xaxis_title="Vehicle Class",
        yaxis_title="Probability",
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# --------------------------------------------------------------------------------
# 4. MAIN APP LAYOUT
# --------------------------------------------------------------------------------
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3097/3097180.png", width=100)
        st.title("Navigation")
        app_mode = st.radio("Go to", ["Classifier", "About Project"])
        st.info("This app uses **EfficientNetB0** trained on a Kaggle Vehicle dataset.")

    if app_mode == "About Project":
        st.title("🚘 Vehicle Image Classification System")
        st.subheader("A Deep Learning–Driven Visual Recognition Pipeline")

        st.markdown(
            """
            This application demonstrates a **production-grade computer vision system**
            that performs **semantic vehicle classification** using a pretrained
            **EfficientNet architecture** fine-tuned on a domain-specific dataset.
            """
        )

        with st.expander("📘 Project Overview", expanded=True):
            st.markdown(
                """
                ### 🔬 Problem Definition
                Vehicle classification is a **fine-grained visual recognition problem**
                involving substantial **intra-class variation**, diverse viewpoints,
                and complex backgrounds.

                ### 🧠 Methodology
                A **transfer learning strategy** is employed using **EfficientNetB0**
                pretrained on ImageNet. The pretrained backbone acts as a robust
                feature extractor, while a custom classification head learns
                task-specific decision boundaries.

                ### 🚘 Target Classes
                - Auto Rickshaws
                - Bikes
                - Cars
                - Motorcycles
                - Planes
                - Ships
                - Trains

                
                ### 🎯 Final Results (Authoritative)
                🔹 **Training Set**

                **Final Training Accuracy:** 99.11%

                **Final Training Loss:** 0.0316

                🔹 **Validation Set**

                **Final Validation Accuracy:** 99.55%

                **Final Validation Loss:** 0.0180
                """
            )


        with st.expander("🧠 Model Architecture & Training Strategy"):
            st.markdown(
                """
                **Backbone Network**
                - EfficientNetB0 with compound scaling
                - Pretrained on ImageNet (1.2M images)

                **Custom Head**
                - Global Average Pooling
                - Batch Normalization
                - Fully Connected Dense Layer (ReLU)
                - Dropout Regularization (p = 0.5)
                - Softmax output layer

                **Optimization**
                - Adam Optimizer
                - Categorical Cross-Entropy Loss
                - Learning Rate Scheduling
                - Early Stopping for generalization control
                """
            )

        st.markdown(
            """
            <style>
            .footer {
                width: 100%;
                text-align: center;
                padding: 20px;
                margin-top: 50px;
                border-top: 1px solid #444;
                color: #888;
                font-size: 14px;
            }
            .footer a {
                color: #FF4B4B;
                text-decoration: none;
                margin: 0 10px;
                font-weight: bold;
            }
            .footer a:hover {
                text-decoration: underline;
            }
            </style>
            
            <div class="footer">
                Developed with ❤️ by <b>Sayambar Roy Chowdhury</b><br>
                <a href="https://github.com/Sayambar2004" target="_blank">GitHub</a> | 
                <a href="https://www.linkedin.com/in/sayambar-roy-chowdhury-731b0a282/" target="_blank">LinkedIn</a>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif app_mode == "Classifier":
        st.title("🚀 Vehicle Classifier Pro")
        st.write("Upload an image of a vehicle, and the our CNN will identify it.")

        # File Uploader
        file = st.file_uploader("Drop your image here...", type=["jpg", "png", "jpeg"])

        if file is not None:
            # Layout: Two columns (Image | Results)
            col1, col2 = st.columns([1, 1.5])

            with col1:
                st.write("### Source Image")
                image = Image.open(file).convert('RGB')
                st.image(image, use_container_width=True, caption="Uploaded Image")

            with col2:
                st.write("### Analysis Results")
                
                # Load model
                model = load_trained_model()
                
                if model:
                    # Progress bar animation for "Thinking" effect
                    with st.spinner("Analyzing pixels..."):
                        # Preprocess
                        processed_img = preprocess_image(image)
                        time.sleep(0.5) # Slight delay for UX effect
                        
                        # Predict
                        predictions = model.predict(processed_img)
                        score = tf.nn.softmax(predictions[0]) # If your model output isn't already softmaxed
                        # Since EfficientNet usually outputs logits or raw probs depending on the final layer.
                        # If your notebook ended with Dense(7, activation='softmax'), predictions is already probs.
                        # Assuming notebook used softmax:
                        confidence_scores = predictions[0]
                        predicted_class_idx = np.argmax(confidence_scores)
                        predicted_class = CLASS_NAMES[predicted_class_idx]
                        confidence = confidence_scores[predicted_class_idx]

                    # Display Top Prediction using Metrics
                    st.success("Analysis Complete!")
                    
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric(label="Predicted Class", value=f"{predicted_class}")
                    with metric_col2:
                        # Color logic for confidence
                        delta_color = "normal"
                        if confidence > 0.90: delta_color = "normal" # Greenish in Streamlit
                        elif confidence < 0.50: delta_color = "inverse" # Red
                        
                        st.metric(label="Confidence Score", value=f"{confidence:.2%}", delta="High Confidence" if confidence > 0.8 else None, delta_color=delta_color)

                    # Plot Chart
                    st.plotly_chart(plot_confidence_chart(predictions), use_container_width=True)

if __name__ == "__main__":
    main()