import streamlit as st
import pandas as pd
import json, tempfile, gc, io
from pdf2image import convert_from_bytes
import google.generativeai as genai
import fitz  # PyMuPDF

# --- 1. SESSION STATE INITIALIZATION ---
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()

# --- 2. OPTIMIZED PROCESSING FUNCTION ---
def process_single_flyer(api_key, uploaded_file, shop_name):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")
    
    extracted_data = []
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    progress_bar = st.progress(0)
    
    for i in range(len(doc)):
        # Convert 1 page at a time to save memory
        images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1, dpi=150)
        
        if images:
            prompt = "Extract product data JSON list: name, std_price, disc_price, unit_price, disc_pct."
            try:
                response = model.generate_content([prompt, images[0]])
                clean_json = response.text.strip().replace("```json", "").replace("```", "")
                page_items = json.loads(clean_json)
                for item in page_items:
                    item["store"] = shop_name
                    extracted_data.append(item)
            except:
                continue
            
            # Memory Cleanup
            del images
            gc.collect()
        
        progress_bar.progress((i + 1) / len(doc))
    
    doc.close()
    return pd.DataFrame(extracted_data)

# --- 3. UI & DOWNLOAD LOGIC ---
st.title("🛒 LT Retail Strategy: Multi-Shop Intelligence")

with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password")
    shop_choice = st.selectbox("Select Shop", ["Lidl", "Maxima", "IKI", "Rimi", "Norfa"])
    uploaded_file = st.file_uploader(f"Upload {shop_choice} Flyer", type="pdf")
    
    if st.button("🚀 Process & Append to Master"):
        if api_key and uploaded_file:
            new_df = process_single_flyer(api_key, uploaded_file, shop_choice)
            # Append new data to the master dataframe
            st.session_state['master_df'] = pd.concat(
                [st.session_state['master_df'], new_df], 
                ignore_index=True
            ).drop_duplicates()
            st.success(f"Added {len(new_df)} products from {shop_choice}!")

# --- 4. DATA DASHBOARD & CSV EXPORT ---
if not st.session_state['master_df'].empty:
    df = st.session_state['master_df']
    
    # Summary Metrics
    c1, c2 = st.columns(2)
    c1.metric("Total Products", len(df))
    c2.metric("Shops in Database", df['store'].nunique())

    # The Download Button
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df(df)
    
    st.download_button(
        label="📥 Download Combined CSV (All Shops)",
        data=csv_data,
        file_name=f"Lithuania_Market_Prices_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.subheader("🔍 Master Table View")
    st.dataframe(df, use_container_width=True)

    if st.button("🗑️ Reset Master Database"):
        st.session_state['master_df'] = pd.DataFrame()
        st.rerun()
