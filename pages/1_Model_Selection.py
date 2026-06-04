import streamlit as st

st.set_page_config(
    page_title="Model Selection",
    layout="wide"
)

# --------------------------------
# CSS
# --------------------------------

st.markdown("""
<style>

div.stButton > button {
    width: 100%;
    height: 80px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 15px;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# Header
# --------------------------------

st.markdown(
    '<p class="title">Choose Analysis Type</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Select the AI model you want to use.</p>',
    unsafe_allow_html=True
)

st.info(
    "Disclaimer: This application is an academic project and should not be used as a substitute for professional medical diagnosis."
)

st.markdown("---")

# --------------------------------
# Cards
# --------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Brain Tumor Detection")

    st.write("""
    Detect:

    • Glioma Tumor

    • Meningioma Tumor

    • Pituitary Tumor

    • No Tumor
    """)

    if st.button(
        "Open Brain Tumor Module",
        key="tumor",
        use_container_width=True
    ):
        st.switch_page(
            "pages/2_Brain_Tumor.py"
        )

with col2:

    st.subheader("Alzheimer's Disease Detection")

    st.write("""
    Detect:

    • No Impairment

    • Very Mild Impairment

    • Mild Impairment

    • Moderate Impairment
    """)

    if st.button(
        "Open Alzheimer's Module",
        key="alz",
        use_container_width=True
    ):
        st.switch_page(
            "pages/3_Alzhimers.py"
        )

st.markdown("---")

# --------------------------------
# Information
# --------------------------------

st.header("About Brain Tumor Detection")

st.write("""
The Brain Tumor module uses MRI scans and a
ResNet18 deep learning model to classify
brain tumors into four categories and
generate Grad-CAM heatmaps for visual
explanation.
""")

st.markdown("---")

st.header("About Alzheimer's Detection")

st.write("""
The Alzheimer's module analyzes MRI scans
to identify different stages of cognitive
impairment using deep learning and
Explainable AI techniques.
""")

st.markdown("---")

if st.button(
    "Back To Home",
    use_container_width=True
):
    st.switch_page("app.py")