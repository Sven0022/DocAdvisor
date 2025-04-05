import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline

st.set_page_config(page_title="DocAdvisor", layout="wide")
st.title("📄 DocAdvisor – AI-Powered Report Fixer")

@st.cache_resource
def load_corrector():
    hf_token = st.secrets["hf_token"]  
    model_id = "shrifhesiub/DocAdvisor" 

    model = BartForConditionalGeneration.from_pretrained(model_id, use_auth_token=hf_token)
    tokenizer = BartTokenizer.from_pretrained(model_id, use_auth_token=hf_token)

    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

corrector = load_corrector()

st.markdown("Paste your business report section below:")

user_input = st.text_area("Text Input", height=250)

if st.button("🔧 Fix My Report"):
    if user_input.strip():
        with st.spinner("Correcting grammar, spelling, and clarity..."):
            result = corrector(user_input, max_length=1024, num_return_sequences=1)[0]['generated_text']
        st.subheader("✅ Enhanced Output:")
        st.success(result)
    else:
        st.warning("Please enter some text to correct.")
