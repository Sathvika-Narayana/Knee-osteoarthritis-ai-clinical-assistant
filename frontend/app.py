import streamlit as st
import requests
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Knee Osteoarthritis AI Assistant",
    layout="wide"
)

st.title("🧠 Knee Osteoarthritis AI Clinical Assistant")

st.markdown("---")

# --------------------------------------------------
# UPLOAD IMAGE
# --------------------------------------------------

uploaded = st.file_uploader(
    "Upload Knee X-ray Image",
    type=["jpg","jpeg","png"]
)

# --------------------------------------------------
# MAIN UI
# --------------------------------------------------

if uploaded:

    image = Image.open(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, width=300)

    # --------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------

    if st.button("🚀 Analyze"):

        with st.spinner("AI analyzing clinical image..."):

            files = {
                "file": (uploaded.name, uploaded.getvalue(), "image/jpeg")
            }

            # Model switching handled only in backend
            data = {
                "model_type": "resnet"
            }

            response = requests.post(
                "http://localhost:8000/predict",
                files=files,
                data=data
            )

        # --------------------------------------------------
        # ERROR HANDLING
        # --------------------------------------------------

        if response.status_code != 200:

            st.error("❌ Backend Processing Error")
            st.write(response.text)
            st.stop()

        result = response.json()

        prediction = result.get("prediction","Unknown")
        gradcam = result.get("gradcam_image",None)
        explanation = result.get("explanation","AI explanation unavailable")

        # --------------------------------------------------
        # CLINICAL MAPPING
        # --------------------------------------------------

        severity_map = {
            "KL0": ("No Osteoarthritis","🟢 Normal","Preventive monitoring"),
            "KL1": ("Doubtful OA","🟢 Mild","Lifestyle management"),
            "KL2": ("Mild OA","🟡 Early Degeneration","Physiotherapy recommended"),
            "KL3": ("Moderate OA","🟠 Progressive","Medication + therapy"),
            "KL4": ("Severe OA","🔴 Advanced","Surgical evaluation suggested")
        }

        diagnosis, risk_text, recommendation = severity_map.get(
            prediction,
            ("Unknown","Unknown","Consult specialist")
        )

        # --------------------------------------------------
        # CLINICAL REPORT HEADER
        # --------------------------------------------------

        st.markdown("## 🏥 KOA Clinical Assessment Report")

        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Diagnosis", prediction)

        with colB:
            st.metric("Clinical Stage", diagnosis)

        with colC:
            st.metric("Risk Level", risk_text)

        st.info(f"Recommended Care: **{recommendation}**")

        # --------------------------------------------------
        # HEATMAP DISPLAY
        # --------------------------------------------------

        if gradcam:
            with col2:
                st.subheader("🔥 GradCAM Visualization")
                st.image(gradcam, width=350)

        # --------------------------------------------------
        # CLINICAL EXPLANATION (GEMINI OUTPUT)
        # --------------------------------------------------

        st.markdown("---")
        st.subheader("📋 AI Clinical Interpretation")

        with st.expander("🧠 View Detailed Clinical Explanation"):

            # Clean Gemini formatting
            explanation = explanation.replace("•","")
            explanation = explanation.replace("*","")

            paragraphs = explanation.split("\n\n")

            for para in paragraphs:
                if para.strip():
                    st.write(para.strip())

        # --------------------------------------------------
        # DISCLAIMER
        # --------------------------------------------------

        st.markdown("---")
        st.caption("⚠️ AI-generated clinical assistance. Not a medical diagnosis.")