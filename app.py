import streamlit as st
import pandas as pd
import plotly.express as px
import json, gc, io, itertools
from pdf2image import convert_from_bytes
import google.generativeai as genai
import fitz 
from rapidfuzz import process, fuzz

st.set_page_config(page_title="LT Retail Strategist Pro", layout="wide", initial_sidebar_state="expanded")

if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()

def process_flyer(api_key, uploaded_file, shop_name):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")
    extracted = []
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    progress = st.progress(0)
    for i in range(len(doc)):
        images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1, dpi=150)
        if images:
            # Force the model to use our specific keys
            prompt = "Extract all products. Return ONLY a JSON list with these keys: 'product_name', 'std_price', 'disc_price', 'unit_price', 'disc_pct'."
            try:
                response = model.generate_content([prompt, images[0]])
                clean = response.text.strip().replace("```json", "").replace("```", "")
                page_items = json.loads(clean)
                
                for item in page_items:
                    # NORMALIZE: Ensure the store is added and keys exist
                    normalized_item = {
                        "product_name": item.get("product_name") or item.get("name") or "Unknown Product",
                        "std_price": item.get("std_price") or 0.0,
                        "disc_price": item.get("disc_price") or item.get("price") or 0.0,
                        "unit_price": item.get("unit_price") or "",
                        "disc_pct": item.get("disc_pct") or 0,
                        "store": shop_name
                    }
                    extracted.append(normalized_item)
            except: continue
            del images
            gc.collect()
        progress.progress((i + 1) / len(doc))
    return pd.DataFrame(extracted)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔑 Authentication")
    api_key = st.text_input("Gemini API Key", type="password")
    st.divider()
    shop = st.selectbox("Select Retailer", ["Lidl", "Maxima", "IKI", "Rimi", "Norfa"])
    file = st.file_uploader(f"Upload {shop} PDF", type="pdf")
    
    if st.button("🚀 Process & Append"):
        if api_key and file:
            new_data = process_flyer(api_key, file, shop)
            if not new_data.empty:
                st.session_state['master_df'] = pd.concat([st.session_state['master_df'], new_data], ignore_index=True).drop_duplicates()
                st.success(f"Captured {len(new_data)} items!")
            else:
                st.error("AI could not find products. Try a different page or check your API key.")

# --- MAIN DASHBOARD ---
if not st.session_state['master_df'].empty:
    df = st.session_state['master_df']
    
    # SAFETY CHECK: Ensure the column exists before calling it
    if 'product_name' in df.columns:
        tab1, tab2 = st.tabs(["📊 Price Analytics", "🍳 AI Recipe Optimizer"])
        
        with tab1:
            st.subheader("Market Comparison")
            item_to_check = st.selectbox("Compare Price of:", sorted(df['product_name'].unique()))
            
            # Filter and Chart
            compare_df = df[df['product_name'] == item_to_check]
            if not compare_df.empty:
                fig = px.bar(compare_df, x='store', y='disc_price', color='store', text='disc_price',
                             labels={'disc_price': 'Price (€)', 'store': 'Shop'})
                st.plotly_chart(fig, use_container_width=True)
            
            st.download_button("📥 Download Combined CSV", df.to_csv(index=False).encode('utf-8'), "market_intel.csv", "text/csv")
    else:
        st.error("Data error: 'product_name' column missing. Please reset and re-upload.")
        if st.button("Reset Data"):
            st.session_state['master_df'] = pd.DataFrame()
            st.rerun()
