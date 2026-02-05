import streamlit as st
import json, tempfile, gc, os
import pandas as pd
from pdf2image import convert_from_bytes
import google.generativeai as genai
import fitz  # PyMuPDF

# --- 1. MEMORY CONFIG ---
# Set the maximum upload to 500MB, but we will process it intelligently.
st.set_page_config(page_title="LT Price Strategist (Safe Mode)", layout="wide")

def process_flyer_safely(api_key, uploaded_file, shop_name):
    """Processes a PDF one page at a time to stay under the 1GB RAM limit."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")
    
    extracted_data = []
    
    # Open PDF with fitz to get the page count without loading images
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(total_pages):
        status_text.text(f"Processing {shop_name}: Page {i+1}/{total_pages}...")
        
        # 🟢 THE TRICK: Convert ONLY the current page (i+1) to an image
        # This prevents the "Memory Explosion" of converting 50 pages at once.
        images = convert_from_bytes(
            pdf_bytes, 
            first_page=i+1, 
            last_page=i+1, 
            dpi=150 # 150 is the optimal balance for Gemini 3
        )
        
        if images:
            page_img = images[0]
            
            prompt = "Extract all product data into a JSON list: name, std_price, disc_price, unit_price, disc_pct."
            try:
                response = model.generate_content([prompt, page_img])
                # Clean and parse JSON
                raw_text = response.text.strip().replace("```json", "").replace("```", "")
                page_items = json.loads(raw_text)
                
                for item in page_items:
                    item["store"] = shop_name
                    extracted_data.append(item)
            except Exception as e:
                st.warning(f"Skipped page {i+1} due to error.")

            # 🔴 CRITICAL: Force clear the image from RAM immediately
            del page_img
            del images
            gc.collect() 
        
        progress_bar.progress((i + 1) / total_pages)

    doc.close()
    return extracted_data

# --- 2. UPDATED UI ---
st.title("🛒 Safe-Mode Market Intelligence")
st.info("This version processes pages one-by-one to prevent Streamlit Cloud crashes.")

with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password")
    st.divider()
    # To save more memory, we'll process shops sequentially
    shop_choice = st.selectbox("Select Shop to Process", ["Lidl", "Maxima", "IKI", "Rimi", "Norfa"])
    uploaded_file = st.file_uploader(f"Upload {shop_choice} Flyer", type="pdf")
    
    if st.button("🚀 Process Current Flyer"):
        if api_key and uploaded_file:
            # Initialize master list in session state if not present
            if 'master_df' not in st.session_state:
                st.session_state['master_df'] = pd.DataFrame()
                
            new_items = process_flyer_safely(api_key, uploaded_file, shop_choice)
            new_df = pd.DataFrame(new_items)
            
            # Append to the existing database
            st.session_state['master_df'] = pd.concat([st.session_state['master_df'], new_df], ignore_index=True)
            st.success(f"Added {len(new_df)} products from {shop_choice}!")

# --- 3. ANALYTICS ---
if 'master_df' in st.session_state and not st.session_state['master_df'].empty:
    df = st.session_state['master_df']
    
    st.subheader("📊 Combined Market Data")
    st.dataframe(df, use_container_width=True)
    
    if st.button("Clear All Data"):
        st.session_state['master_df'] = pd.DataFrame()
        st.rerun()
