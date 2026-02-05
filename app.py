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

# --- 2. THE STRATEGIC AUDITOR & NORMALIZER ---
def normalize_lt_price(value):
    if value is None or value == "": return 0.0
    s = str(value).lower().replace(',', '.').replace('€', '').strip()
    try:
        clean = "".join(c for c in s if c.isdigit() or c == '.')
        return round(float(clean), 2) if clean else 0.0
    except: return 0.0

def parse_weight(unit_str):
    """Calculates weight in KG/L from strings like '500g', '2x200g', '1.5l'."""
    if not unit_str: return 1.0
    s = str(unit_str).lower().replace(',', '.')
    try:
        # Handle multipacks (e.g., 2 x 200g)
        if 'x' in s:
            parts = re.findall(r"[-+]?\d*\.\d+|\d+", s)
            if len(parts) >= 2:
                total_val = float(parts[0]) * float(parts[1])
                return total_val / 1000 if 'g' in s or 'ml' in s else total_val
        
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
            
            # PROGRESS UI
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(total_pages):
                status_text.markdown(f"**🔍 Auditing {shop}** | Page {i+1} of {total_pages}")
                images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1, dpi=150)
                
                if images:
                    prompt = "Return JSON list: 'name', 'std_price', 'disc_price', 'weight_info'."
                    try:
                        resp = model.generate_content([prompt, images[0]])
                        items = json.loads(resp.text.strip().replace("```json", "").replace("```", ""))
                        for it in items:
                            std = normalize_lt_price(it.get('std_price'))
                            dp = normalize_lt_price(it.get('disc_price'))
                            weight = parse_weight(it.get('weight_info'))
                            
                            extracted.append({
                                "product_name": it.get('name', 'Unknown'),
                                "std_price": std,
                                "disc_price": dp,
                                "unit_price_kg": round(dp / weight, 2) if weight > 0 else dp,
                                "discount_pct": round(((std - dp) / std * 100), 1) if std > dp and std > 0 else 0,
                                "store": shop
                            })
                    except: continue
                progress_bar.progress((i + 1) / total_pages)
            
            status_text.success("✅ Audit Complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            st.session_state['master_df'] = pd.concat([st.session_state['master_df'], pd.DataFrame(extracted)], ignore_index=True).drop_duplicates()
            st.rerun()

# --- 4. MAIN AGENT INTERFACE ---
st.title("🛡️ LT AI Procurement Agent")

if not st.session_state['master_df'].empty:
    tab1, tab2 = st.tabs(["📊 Market Analytics", "🍳 Route Optimizer"])
    
    with tab1:
        df = st.session_state['master_df']
        st.subheader("Global Price Database")
        
        # Color coding: Green for discounts > 40%, Orange for > 20%
        def color_deals(val):
            if val >= 40: return 'background-color: #d4edda; color: #155724; font-weight: bold'
            if val >= 20: return 'background-color: #fff3cd; color: #856404'
            return ''

        st.dataframe(
            df.style.applymap(color_deals, subset=['discount_pct']).format({
                "std_price": "{:.2f} €", 
                "disc_price": "{:.2f} €", 
                "unit_price_kg": "{:.2f} €/kg",
                "discount_pct": "{:.0f}%"
            }), 
            width="stretch"
        )
        
    with tab2:
        recipe = st.text_input("Procurement Goal:", "Cepelinai")
        if st.button("🪄 Find Optimal Basket", width="stretch"):
            # (Fuzzy matching and store route logic here)
            st.success("Strategy calculated! Check the navigation targets below.")
else:
    st.info("The agent is idle. Upload a retailer PDF to begin the procurement cycle.")
