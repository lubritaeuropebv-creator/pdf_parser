import streamlit as st
import pandas as pd
import plotly.express as px
import json, gc, re, time
from pdf2image import convert_from_bytes
import google.generativeai as genai
import fitz 
from rapidfuzz import process, fuzz

# --- 1. CONFIG & AGENT SETUP ---
st.set_page_config(page_title="LT AI Procurement Agent", layout="wide")

if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()

# --- 2. LITHUANIAN RETAIL NORMALIZER ---
def normalize_lt_price(value):
    if value is None or value == "": return 0.0
    s = str(value).lower().replace(',', '.').replace('€', '').strip()
    try:
        clean = "".join(c for c in s if c.isdigit() or c == '.')
        return round(float(clean), 2) if clean else 0.0
    except: return 0.0

def parse_weight(unit_str):
    if not unit_str: return 1.0
    s = str(unit_str).lower().replace(',', '.')
    try:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        val = float(nums[0])
        if 'g' in s or 'ml' in s: return val / 1000
        return val
    except: return 1.0

# --- 3. SIDEBAR: DATA INGESTION WITH PROGRESS ---
with st.sidebar:
    st.header("🔑 Authenticate")
    api_key = st.text_input("Gemini API Key", type="password")
    st.divider()
    shop = st.selectbox("Current Retailer", ["Lidl", "Maxima", "IKI", "Rimi", "Norfa"])
    file = st.file_uploader(f"Upload {shop} PDF", type="pdf")
    
    if st.button("🚀 Execute Scrape & Audit", width="stretch"):
        if api_key and file:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-3-flash-preview")
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            extracted = []
            
            # INITIALIZE PROGRESS ELEMENTS
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(total_pages):
                # Update visual progress
                status_text.text(f"🔍 Analyzing {shop} - Page {i+1} of {total_pages}")
                
                images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1, dpi=150)
                if images:
                    prompt = "Return JSON list of products with: 'name', 'std_price', 'disc_price', 'weight'."
                    try:
                        resp = model.generate_content([prompt, images[0]])
                        items = json.loads(resp.text.strip().replace("```json", "").replace("```", ""))
                        for it in items:
                            dp = normalize_lt_price(it.get('disc_price'))
                            w = parse_weight(it.get('weight'))
                            extracted.append({
                                "product_name": it.get('name', 'Unknown'),
                                "std_price": normalize_lt_price(it.get('std_price')),
                                "disc_price": dp,
                                "unit_price_kg": round(dp / w, 2) if w > 0 else dp,
                                "store": shop
                            })
                    except Exception as e:
                        st.warning(f"Skipped page {i+1} due to error.")
                
                # Update progress bar percentage
                progress_bar.progress((i + 1) / total_pages)
            
            status_text.text("✅ Audit Complete! Updating Database...")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            st.session_state['master_df'] = pd.concat([st.session_state['master_df'], pd.DataFrame(extracted)], ignore_index=True).drop_duplicates()
            st.rerun()

# --- 4. MAIN INTERFACE ---
st.title("🛡️ LT AI Procurement Agent")

if not st.session_state['master_df'].empty:
    tab1, tab2 = st.tabs(["📊 Market Analytics", "🍳 Route Optimizer"])
    
    with tab1:
        df = st.session_state['master_df']
        st.subheader("Global Price Database")
        st.dataframe(df.style.format({"std_price": "{:.2f} €", "disc_price": "{:.2f} €", "unit_price_kg": "{:.2f} €/kg"}), width="stretch")
        
    with tab2:
        recipe = st.text_input("Procurement Goal:", "Cepelinai")
        if st.button("🪄 Calculate Optimal Route", width="stretch"):
            with st.spinner("Agent is calculating market routes..."):
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-3-flash-preview")
                
                # AI Decomposition
                ings_resp = model.generate_content(f"List 5 main ingredients for {recipe} in Lithuanian. Return only as comma-separated list.")
                ings = ings_resp.text.strip().split(',')
                
                # Fuzzy Sourcing
                basket = []
                for ing in ings:
                    match = process.extractOne(ing.strip(), df['product_name'], scorer=fuzz.WRatio)
                    if match and match[1] > 60:
                        basket.append(df.iloc[match[2]])
                
                if basket:
                    basket_df = pd.DataFrame(basket)
                    st.success(f"Optimized Strategy Found for {recipe}!")
                    st.dataframe(basket_df, width="stretch")
                    
                    top_stores = basket_df['store'].unique()[:2]
                    st.subheader(f"📍 Optimal Route: {' ➔ '.join(top_stores)}")
                    
                    # Google Maps Logic
                    map_url = f"https://www.google.com/maps/dir/?api=1&destination={top_stores[-1]}&waypoints={top_stores[0]}"
                    st.link_button("🗺️ Start Navigation", map_url)
                else:
                    st.error("No ingredients found in current data. Upload more flyers!")
else:
    st.info("The agent is idle. Please upload a retailer PDF in the sidebar.")
