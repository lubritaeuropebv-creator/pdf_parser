import streamlit as st
import pandas as pd
import plotly.express as px
import json, gc, re, urllib.parse
from pdf2image import convert_from_bytes
import google.generativeai as genai
import fitz 
from rapidfuzz import process, fuzz

# --- 1. CORE CONFIG & AGENT SETUP ---
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

# --- 3. SIDEBAR: DATA INGESTION ---
with st.sidebar:
    st.header("🔑 Authenticate")
    api_key = st.text_input("Gemini API Key", type="password")
    st.divider()
    shop = st.selectbox("Current Retailer", ["Lidl", "Maxima", "IKI", "Rimi", "Norfa"])
    file = st.file_uploader(f"Upload {shop} PDF", type="pdf")
    
    if st.button("🚀 Scrape & Audit", width="stretch"):
        if api_key and file:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-3-flash-preview")
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            extracted = []
            
            for i in range(len(doc)):
                images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1, dpi=150)
                if images:
                    prompt = "Return JSON list: 'name', 'std_price', 'disc_price', 'weight'."
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
            st.session_state['master_df'] = pd.concat([st.session_state['master_df'], pd.DataFrame(extracted)], ignore_index=True).drop_duplicates()
            st.rerun()

# --- 4. MAIN INTERFACE ---
st.title("🛡️ LT AI Procurement Agent")

if not st.session_state['master_df'].empty:
    tab1, tab2 = st.tabs(["📊 Market Analytics", "🍳 Route Optimizer"])
    
    with tab1:
        df = st.session_state['master_df']
        st.dataframe(df, width="stretch")
        
    with tab2:
        recipe = st.text_input("Procurement Goal:", "Cepelinai")
        if st.button("🪄 Calculate Optimal Route", width="stretch"):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-3-flash-preview")
            
            # Step 1: AI Decomposition
            ings = model.generate_content(f"5 ingredients for {recipe} in LT. CSV only.").text.strip().split(',')
            
            # Step 2: Fuzzy Sourcing
            basket = []
            for ing in ings:
                match = process.extractOne(ing.strip(), df['product_name'], scorer=fuzz.WRatio)
                if match and match[1] > 60:
                    basket.append(df.iloc[match[2]])
            
            if basket:
                basket_df = pd.DataFrame(basket)
                st.dataframe(basket_df, width="stretch")
                
                # Step 3: Route Generation
                top_stores = basket_df['store'].unique()[:2] # Top 2 stops
                st.subheader(f"📍 Optimal Route: {' ➔ '.join(top_stores)}")
                
                # Create Google Maps Link
                origin = "Vilnius" # Or user current loc
                stops = f"{top_stores[0]}+{top_stores[1]}" if len(top_stores)>1 else top_stores[0]
                map_url = f"https://www.google.com/maps/dir/?api=1&destination={top_stores[-1]}&waypoints={top_stores[0]}"
                st.link_button("🗺️ Open Navigation", map_url)
            else:
                st.error("No ingredients found. Upload more flyers!")
else:
    st.info("The agent is idle. Upload a retailer PDF to begin.")
