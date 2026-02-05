import streamlit as st
import pandas as pd
import plotly.express as px
import json, gc, io, itertools
from pdf2image import convert_from_bytes
import google.generativeai as genai
import fitz 
from rapidfuzz import process, fuzz

# --- 1. CONFIG (2026 STANDARDS) ---
st.set_page_config(
    page_title="LT Price Strategist v2.0", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()

# --- 2. LITHUANIAN RETAIL NORMALIZER ---
def normalize_lt_price(value):
    """
    Cleans strings like '1,99 €', '2 vnt. už 3.00', or '0.99/kg' into clean floats.
    """
    if value is None or value == "": return 0.0
    if isinstance(value, (int, float)): return float(value)
    
    # 1. Standardize formatting (commas to dots)
    s = str(value).lower().replace(',', '.').replace('€', '').strip()
    
    # 2. Logic for "Buy 2 for X" - extract the total and divide if needed
    # Example: "2 vnt - 3.00" -> 1.50
    if "už" in s or "vnt" in s:
        try:
            parts = [float(x) for x in "".join(c if c.isdigit() or c == '.' else ' ' for c in s).split()]
            if len(parts) >= 2:
                return round(parts[1] / parts[0], 2)
            elif len(parts) == 1:
                return parts[0]
        except: pass

    # 3. Clean remaining text and symbols
    try:
        clean_val = "".join(c for c in s if c.isdigit() or c == '.')
        return round(float(clean_val), 2) if clean_val else 0.0
    except:
        return 0.0

# --- 3. PROCESSING ENGINE ---
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
            prompt = """Extract product data. 
            Return JSON list: 'product_name', 'std_price', 'disc_price', 'unit_price', 'disc_pct'. 
            If price is 'Buy 2 for X', put the raw string in 'disc_price'."""
            try:
                response = model.generate_content([prompt, images[0]])
                clean = response.text.strip().replace("```json", "").replace("```", "")
                page_items = json.loads(clean)
                for item in page_items:
                    # Apply Lithuanian Normalizer
                    extracted_data.append({
                        "product_name": str(item.get("product_name") or "Unknown"),
                        "std_price": normalize_lt_price(item.get("std_price")),
                        "disc_price": normalize_lt_price(item.get("disc_price")),
                        "unit_price": str(item.get("unit_price") or ""),
                        "store": str(shop_name)
                    })
            except: continue
            del images
            gc.collect()
        progress_bar.progress((i + 1) / len(doc))
    return pd.DataFrame(extracted_data)

# --- 4. UI DASHBOARD ---
with st.sidebar:
    st.header("🔑 Authenticate")
    api_key = st.text_input("Gemini API Key", type="password")
    
    if not st.session_state['master_df'].empty:
        st.divider()
        st.subheader("📥 Export Hub")
        csv = st.session_state['master_df'].to_csv(index=False).encode('utf-8')
        st.download_button("Download Strategy CSV", csv, "prices.csv", "text/csv")
        if st.button("🗑️ Reset All"):
            st.session_state['master_df'] = pd.DataFrame()
            st.rerun()

    st.divider()
    shop = st.selectbox("Retailer", ["Lidl", "Maxima", "IKI", "Rimi", "Norfa"])
    file = st.file_uploader(f"Upload {shop} PDF", type="pdf")
    
    if st.button("🚀 Scrape & Normalize"):
        if api_key and file:
            new_data = process_flyer(api_key, file, shop)
            st.session_state['master_df'] = pd.concat([st.session_state['master_df'], new_data], ignore_index=True).drop_duplicates()
            st.rerun()

st.title("🛒 LT Retail Strategy Engine")

if not st.session_state['master_df'].empty:
    df = st.session_state['master_df']
    
    # Display the Cleaned Data with 2 decimal precision
    st.subheader("Live Market Intel")
    st.dataframe(df.style.format({"std_price": "{:.2f} €", "disc_price": "{:.2f} €"}), width="stretch")
    
    # Analytics
    search = st.selectbox("Compare Specific Item:", sorted(df['product_name'].unique()))
    comp_df = df[df['product_name'] == search]
    fig = px.bar(comp_df, x='store', y='disc_price', color='store', text_auto='.2f', title=f"Price Wars: {search}")
    st.plotly_chart(fig, width="stretch")
else:
    st.info("👈 Upload your first flyer in the sidebar to begin price analysis.")
