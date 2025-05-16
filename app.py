import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline

# Page configuration
st.set_page_config(
    page_title="DocAdvisor - AI Report Enhancer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.header("📄 DocAdvisor")
    st.write(
        "Enhance your business report with AI-driven grammar, spelling, and clarity improvements."
    )
    st.markdown("---")
    st.info(
        "Paste your text below and click the button to generate improvements."
    )

# Load model once
@st.cache_resource
def load_corrector():
    hf_token = st.secrets["hf_token"]
    model_id = "shrifhesiub/DocAdvisor"
    model = BartForConditionalGeneration.from_pretrained(
        model_id, use_auth_token=hf_token
    )
    tokenizer = BartTokenizer.from_pretrained(
        model_id, use_auth_token=hf_token
    )
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

corrector = load_corrector()

# Main content area
st.markdown("## Enhance Your Report")
input_col, output_col = st.columns(2)

# Input section
with input_col:
    st.subheader("Your Input")
    user_input = st.text_area(
        "Paste text here...", height=300
    )
    if st.button("Generate Improvements", use_container_width=True):
        if user_input.strip():
            with st.spinner("Working on your report..."):
                result = corrector(
                    user_input, max_length=1024, num_return_sequences=1
                )[0]["generated_text"]
            st.session_state["enhanced_text"] = result
        else:
            st.warning("Please provide some text to enhance.")

# Output section
with output_col:
    st.subheader("Enhanced Output")
    if "enhanced_text" in st.session_state:
        st.success(st.session_state["enhanced_text"])
    else:
        st.write("_Your improved report will appear here._")


