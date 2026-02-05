import streamlit as st
import json, os, datetime, itertools, tempfile, time
import pandas as pd
from pdf2image import convert_from_path
import google.generativeai as genai
from rapidfuzz import process, fuzz
import fitz # PyMuPDF

# --- 1. CORE CONFIGURATION ---
st.set_page_config(page_title="LT Promo Intel Pro (Gemini 3)", layout="wide")

# --- 2. THE UPGRADED EXTRACTION LOGIC ---

def run_extraction_engine(api_key, files_to_process):
    genai.configure(api_key=api_key)
    
    # UPGRADE: Using Gemini 3 Flash for PhD-level reasoning at scale
    model = genai.GenerativeModel(
        model_name="gemini-3-flash-preview", 
        system_instruction=(
            "You are an elite retail data analyst. Your mission is total data capture. "
            "Using Agentic Vision, scan the document for EVERY product, even those in 'fine print' "
            "or corner bubbles. Extract: product_name, category, standard_price, discount_price."
        )
    )

    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Pre-processing setup
    pdf_map = {}
    total_pages = 0
    for file, name in files_to_process:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            # 150 DPI is optimized for Gemini 3's high-fidelity vision
            pages = convert_from_path(tmp.name, dpi=150)
            pdf_map[name] = pages
            total_pages += len(pages)

    start_time = time.time()
    processed_count = 0

    for name, pages in pdf_map.items():
        for i, page_img in enumerate(pages):
            processed_count += 1
            
            # ETR Calculation
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_count
            etr = int(avg_time * (total_pages - processed_count))
            
            status_text.markdown(f"**⚡ Gemini 3 Analyzing {name}:** Page {i+1}/{len(pages)} | ⏳ ETR: {etr}s")
            progress_bar.progress(processed_count / total_pages)

            # AGGRESSIVE EXTRACTION PROMPT
            prompt = """
            Perform a deep-scan of this page. Identify every product and discount.
            Return ONLY a valid JSON list of objects:
            [{"product_name": "string", "category": "string", "standard_price": float, "discount_price": float}]
            If standard_price is missing, estimate it as discount_price / 0.7.
            """
            
            try:
                # Gemini 3 Flash handles images + text with much higher accuracy
                response = model.generate_content([prompt, page_img])
                # Clean and parse JSON
                raw_json = response.text.strip().replace("```json", "").replace("```", "")
                items = json.loads(raw_json)
                for item in items:
                    item["store"] = name
                    item["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d")
                    all_data.append(item)
            except Exception as e:
                continue

    return pd.DataFrame(all_data)

# --- 3. STREAMLIT UI ---

st.title("🛒 LT Promo Intelligence Pro")
st.info("Currently using **Gemini 3 Flash** for Agentic Vision and multimodal data extraction.")

with st.sidebar:
    st.header("1. Authentication")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    st.divider()
    st.header("2. Flyer Upload")
    lidl_pdf = st.file_uploader("Lidl PDF", type="pdf")
    maxima_pdf = st.file_uploader("Maxima PDF", type="pdf")
    process_btn = st.button("🚀 Execute Strategic Analysis", use_container_width=True)

if process_btn:
    if api_key and lidl_pdf and maxima_pdf:
        df = run_extraction_engine(api_key, [(lidl_pdf, "LIDL"), (maxima_pdf, "MAXIMA")])
        st.session_state['df'] = df
        st.success(f"Analysis complete! {len(df)} products captured.")
    else:
        st.warning("Please provide the API key and both flyer PDFs.")

# --- 4. DATA VISUALIZATION & MATCHING ---

if 'df' in st.session_state and not st.session_state['df'].empty:
    df = st.session_state['df']
    
    st.divider()
    st.subheader("📊 Strategic Dashboard")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        search_query = st.text_input("Quick Product Search (Lithuanian)", "pienas")
        if search_query:
            # Fuzzy Matching Logic for Lithuanian Declensions
            matches = process.extract(search_query, df['product_name'].tolist(), scorer=fuzz.WRatio, limit=5)
            match_names = [m[0] for m in matches if m[1] > 60]
            results = df[df['product_name'].isin(match_names)]
            st.dataframe(results[['product_name', 'store', 'discount_price']], use_container_width=True)

    with col2:
        # Aggregated View
        st.write("Full Extracted Database")
        st.dataframe(df, use_container_width=True)
        
    # Download Action
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export to CSV", csv, "flyer_promos.csv", "text/csv")
