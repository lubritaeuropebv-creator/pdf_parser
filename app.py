import streamlit as st
import json, tempfile, time, datetime
import pandas as pd
from pdf2image import convert_from_path
import google.generativeai as genai
import fitz # PyMuPDF

# --- CONFIG ---
st.set_page_config(page_title="Flyer Intelligence Pro", layout="wide")

# --- IMPROVED EXTRACTION FUNCTION ---
def extract_with_gemini_3(api_key, pdf_files):
    genai.configure(api_key=api_key)
    
    # Using the latest Gemini 3 Flash Preview
    model = genai.GenerativeModel(
        model_name="gemini-3-flash-preview",
        system_instruction="You are a retail price extractor. Your goal is 100% recall. List EVERY product visible."
    )
    
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Convert and Count
    pdf_pages = []
    for file, name in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            # 200 DPI is the sweet spot for Gemini 3 Flash's high-res sensors
            images = convert_from_path(tmp.name, dpi=200)
            for i, img in enumerate(images):
                pdf_pages.append({"name": name, "img": img, "page": i+1})
    
    total = len(pdf_pages)
    start_time = time.time()

    # 2. Extract
    for idx, item in enumerate(pdf_pages):
        status_text.write(f"🔍 Analyzing {item['name']} Page {item['page']} of {total}...")
        progress_bar.progress((idx + 1) / total)
        
        # We ask for a simple structure to minimize failures
        prompt = """
        Return a JSON list of ALL products on this page. 
        Include: {"product_name": "...", "discount_price": 0.00}. 
        If you see a price like '1,99€', return 1.99. 
        ONLY return the JSON list. No preamble.
        """
        
        try:
            response = model.generate_content([prompt, item['img']])
            # Robust JSON cleaning
            text_response = response.text.strip()
            if "```json" in text_response:
                text_response = text_response.split("```json")[1].split("```")[0]
            elif "```" in text_response:
                text_response = text_response.split("```")[1].split("```")[0]
            
            page_data = json.loads(text_response)
            for p in page_data:
                p["store"] = item["name"]
                all_data.append(p)
        except Exception as e:
            st.warning(f"Error on {item['name']} P{item['page']}: {str(e)[:50]}")
            continue

    return pd.DataFrame(all_data)

# --- UI ---
st.title("🛒 Promo Intelligence (Gemini 3 Flash)")

with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password")
    lidl = st.file_uploader("Lidl PDF", type="pdf")
    maxima = st.file_uploader("Maxima PDF", type="pdf")
    go = st.button("Start Extraction")

if go and api_key and lidl and maxima:
    results_df = extract_with_gemini_3(api_key, [(lidl, "Lidl"), (maxima, "Maxima")])
    if not results_df.empty:
        st.session_state['df'] = results_df
        st.success(f"Captured {len(results_df)} products!")
    else:
        st.error("Still 0 products. Check if the PDF images are blurry or if your API Key has vision permissions.")

if 'df' in st.session_state:
    st.dataframe(st.session_state['df'], use_container_width=True)
