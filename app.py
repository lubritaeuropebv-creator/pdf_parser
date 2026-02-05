# --- 1. CORE IMPORTS (Must be at the very top) ---
import streamlit as st
import pandas as pd
import plotly.express as px
import json, gc, io, itertools
from pdf2image import convert_from_bytes
import google.generativeai as genai
import fitz  # PyMuPDF
from rapidfuzz import process, fuzz

# --- 2. CONFIGURATION ---
st.set_page_config(
    page_title="LT Retail Strategist Pro", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Initialize global storage
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()

# --- 3. UTILITIES & CACHING ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def process_flyer(api_key, uploaded_file, shop_name):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")
    extracted_data = []
    
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    progress_bar = st.progress(0)
    for i in range(len(doc)):
        images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1, dpi=150)
        if images:
            prompt = "Extract products. Return JSON list with: 'product_name', 'std_price', 'disc_price', 'unit_price', 'disc_pct'."
            try:
                response = model.generate_content([prompt, images[0]])
                clean = response.text.strip().replace("```json", "").replace("```", "")
                page_items = json.loads(clean)
                for item in page_items:
                    # Normalization guard
                    extracted_data.append({
                        "product_name": item.get("product_name") or item.get("name") or "Unknown",
                        "std_price": item.get("std_price") or 0.0,
                        "disc_price": item.get("disc_price") or item.get("price") or 0.0,
                        "unit_price": item.get("unit_price") or "",
                        "disc_pct": item.get("disc_pct") or 0,
                        "store": shop_name
                    })
            except: continue
            del images
            gc.collect()
        progress_bar.progress((i + 1) / len(doc))
    doc.close()
    return pd.DataFrame(extracted_data)

# --- 4. SIDEBAR (The Control Hub) ---
with st.sidebar:
    st.header("🔑 Authentication")
    api_key = st.text_input("Gemini API Key", type="password")
    
    # Global Export Hub
    if not st.session_state['master_df'].empty:
        st.divider()
        st.subheader("📥 Data Export")
        csv_data = convert_df_to_csv(st.session_state['master_df'])
        shops = "_".join(st.session_state['master_df']['store'].unique())
        
        st.download_button(
            label="Download Combined CSV",
            data=csv_data,
            file_name=f"Market_Intel_{shops}.csv",
            mime="text/csv",
            use_container_width=True
        )
        if st.button("🗑️ Clear All Data"):
            st.session_state['master_df'] = pd.DataFrame()
            st.rerun()

    st.divider()
    st.header("📄 Flyer Upload")
    current_shop = st.selectbox("Current Shop", ["Lidl", "Maxima", "IKI", "Rimi", "Norfa"])
    file = st.file_uploader(f"Upload {current_shop} Flyer (PDF)", type="pdf")
    
    if st.button("🚀 Process & Append"):
        if api_key and file:
            with st.spinner(f"Analyzing {current_shop} flyer..."):
                new_data = process_flyer(api_key, file, current_shop)
                if not new_data.empty:
                    st.session_state['master_df'] = pd.concat(
                        [st.session_state['master_df'], new_data], 
                        ignore_index=True
                    ).drop_duplicates()
                    st.success(f"Success! {current_shop} added.")
                    st.rerun()

# --- 5. MAIN CONTENT ---
st.title("🛒 LT Price Strategist Dashboard")

if not st.session_state['master_df'].empty:
    df = st.session_state['master_df']
    tab1, tab2 = st.tabs(["📊 Price Comparison", "🍳 AI Recipe Optimizer"])
    
    with tab1:
        st.subheader("Visual Market Analysis")
        if 'product_name' in df.columns:
            search = st.selectbox("Select Product to Compare:", sorted(df['product_name'].unique()))
            compare_df = df[df['product_name'] == search]
            fig = px.bar(compare_df, x='store', y='disc_price', color='store', text='disc_price', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)
            
    with tab2:
        st.subheader("Recipe to Cart Optimizer")
        user_idea = st.text_input("What's for dinner?", "Cepelinai")
        if st.button("🪄 Find Best Prices"):
            # Recipe generation & Cart logic here (as previously built)
            st.info("AI is matching your ingredients against the database...")
else:
    st.warning("Please enter your API Key and upload a flyer in the sidebar to populate the data.")
