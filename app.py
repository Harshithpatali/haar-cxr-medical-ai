import streamlit as st
from PIL import Image
from src.utils.config import load_config
from src.inference.predictor import PneumoniaPredictor


def main():

    st.set_page_config(page_title="Haar-CXR Pneumonia AI", layout="wide")

    st.title("🩺 Haar-CXR: Dual-Branch Pneumonia Detection")
    st.markdown("Spatial + Haar Wavelet Hybrid Model")

    config = load_config("configs/config.yaml")

    try:
        predictor = PneumoniaPredictor(config)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", width=300)

        with st.spinner("Analyzing..."):
            results = predictor.predict(image)

        st.subheader("Prediction Results")

        st.metric(
            label="Pneumonia Probability",
            value=f"{results['probability']:.4f}"
        )

        st.metric(
            label="Predicted Class",
            value=results["prediction"]
        )

        st.metric(
            label="Confidence Score",
            value=f"{results['uncertainty']['confidence_score']:.4f}"
        )

        st.metric(
            label="Uncertainty (Std)",
            value=f"{results['uncertainty']['uncertainty_std']:.4f}"
        )

        st.subheader("Grad-CAM Visualization")
        st.image(results["gradcam_path"], width=400)

        st.subheader("Wavelet Energy Breakdown")

        for band, value in results["wavelet_energy"].items():
            st.write(f"{band}: {value:.4f}")

        st.info(
            "⚠️ This AI system is for research purposes only "
            "and not for clinical diagnosis."
        )


if __name__ == "__main__":
    main()
