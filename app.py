import streamlit as st
import pandas as pd
import json, gc, re, time
from pdf2image import convert_from_bytes
import google.generativeai as genai
import fitz 
from rapidfuzz import process, fuzz

# --- 1. CONFIG ---
st.set_page_config(
    page_title="LT AI Kitchen Strategist", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()
if 'last_batch_df' not in st.session_state:
    st.session_state['last_batch_df'] = pd.DataFrame()
if 'last_shop_name' not in st.session_state:
    st.session_state['last_shop_name'] = ""

# --- 2. ROBUST PARSING LOGIC ---
def normalize_lt_price(value):
    if value is None or value == "": return 0.0
    s = str(value).lower().replace(',', '.').replace('€', '').strip()
    try:
        clean = re.findall(r"\d+\.\d+|\d+", s)
        return float(clean[0]) if clean else 0.0
    except: return 0.0

def parse_weight(size_str):
    if not size_str: return 0.0
    s = str(size_str).lower().replace(',', '.').strip()
    try:
        multiplier = 1.0
        if 'x' in s:
            parts = s.split('x')
            try:
                multiplier = float(re.findall(r"\d+\.?\d*", parts[0])[0])
                s = parts[1]
            except: pass

        nums = re.findall(r"\d+\.?\d*", s)
        if not nums: return 0.0
        val = float(nums[0]) * multiplier

        if 'mg' in s: return val / 1_000_000
        if 'kg' in s or 'l' in s and 'ml' not in s: return val
        if 'g' in s or 'ml' in s: return val / 1000 
        
        return val / 1000 if val > 20 else val
    except: return 0.0

# --- 3. SIDEBAR: DATA INGESTION ---
with st.sidebar:
    st.header("🔑 Authentication")
    api_key = st.text_input("Gemini API Key", type="password")
    
    if not st.session_state['master_df'].empty:
        st.divider()
        st.subheader("💾 Data Hub")
        csv_master = st.session_state['master_df'].to_csv(index=False).encode('utf-8')
        st.download_button("Download Master DB", csv_master, "LT_Master_Prices.csv", "text/csv", type="primary")
        if st.button("🗑️ Clear Database"):
            st.session_state['master_df'] = pd.DataFrame()
            st.session_state['last_batch_df'] = pd.DataFrame()
            st.rerun()

    st.divider()
    st.header("📄 Upload Flyer")
    shop = st.selectbox("Retailer", ["Lidl", "Maxima", "IKI", "Rimi", "Norfa"])
    file = st.file_uploader(f"Upload {shop} PDF", type="pdf")
    
    if st.button("🚀 Deep Scan & Audit", width="stretch"):
        if api_key and file:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-3-flash-preview")
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            extracted = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(total_pages):
                status_text.markdown(f"**🔍 Page {i+1}/{total_pages}** | Scanning {shop}...")
                images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1, dpi=220)
                
                if images:
                    prompt = """
                    Extract products. JSON keys: 'product_name', 'package_size', 'std_price', 'disc_price'.
                    """
                    try:
                        resp = model.generate_content([prompt, images[0]])
                        clean_json = resp.text.strip().replace("```json", "").replace("```", "")
                        items = json.loads(clean_json)
                        for it in items:
                            std = normalize_lt_price(it.get('std_price'))
                            dp = normalize_lt_price(it.get('disc_price'))
                            if dp == 0 and std > 0: dp = std
                            pkg = str(it.get('package_size', ''))
                            w = parse_weight(pkg)
                            extracted.append({
                                "product_name": it.get('product_name', 'Unknown'),
                                "package_size": pkg,
                                "std_price": std,
                                "disc_price": dp,
                                "unit_price_kg": round(dp / w, 2) if w > 0 else 0.0,
                                "discount_pct": round(((std - dp) / std * 100), 1) if std > dp and std > 0 else 0,
                                "store": shop
                            })
                    except: pass
                progress_bar.progress((i + 1) / total_pages)
            
            status_text.success("✅ Audit Complete!")
            time.sleep(1)
            new_batch_df = pd.DataFrame(extracted)
            st.session_state['last_batch_df'] = new_batch_df
            st.session_state['last_shop_name'] = shop
            st.session_state['master_df'] = pd.concat([st.session_state['master_df'], new_batch_df], ignore_index=True).drop_duplicates()
            st.rerun()

# --- 4. DASHBOARD ---
st.title("🛡️ LT AI Kitchen Strategist")

# Post-Upload Download Alert
if not st.session_state['last_batch_df'].empty:
    with st.expander(f"✅ Processed {st.session_state['last_shop_name']} Data", expanded=True):
        csv_batch = st.session_state['last_batch_df'].to_csv(index=False).encode('utf-8')
        st.download_button(f"Save {st.session_state['last_shop_name']} CSV", csv_batch, "batch.csv")

if not st.session_state['master_df'].empty:
    tab1, tab2 = st.tabs(["📊 Market Analytics", "👨‍🍳 AI Recipe & Basket"])
    
    # --- TAB 1: DATA BROWSER ---
    with tab1:
        df = st.session_state['master_df']
        def highlight_deals(val):
            return 'background-color: #d4edda; color: #155724' if val >= 40 else ''
        st.dataframe(
            df[['product_name', 'package_size', 'disc_price', 'unit_price_kg', 'discount_pct', 'store']]
            .style.applymap(highlight_deals, subset=['discount_pct'])
            .format({"disc_price": "{:.2f}€", "unit_price_kg": "{:.2f}€/kg", "discount_pct": "{:.0f}%"}),
            width="stretch"
        )

    # --- TAB 2: AI RECIPE GENERATOR ---
    with tab2:
        st.markdown("### 💡 From Idea to Optimized Table")
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            user_input = st.text_input("What do you have? Or what do you crave?", placeholder="e.g., 'I have eggs and tomatoes'")
        with col_btn:
            generate_btn = st.button("👨‍🍳 Invent & Shop", type="primary", use_container_width=True)

        if generate_btn and user_input:
            # --- NEW: PERSON COUNT SELECTOR ---
            num_people = st.slider("Number of people to prepare for:", 1, 10, 2)
            
            with st.spinner("🧠 AI Strategist is analyzing market data..."):
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-3-flash-preview")
                
                # Convert flyer database to JSON context
                market_context = st.session_state['master_df'][['product_name', 'package_size', 'disc_price', 'std_price', 'store']].to_json(orient='records')

                # 1. THE STRATEGY PROMPT (Now includes package_size and person scaling)
                strategy_prompt = f"""
                You are an elite grocery price strategist. 
                Task:
                1. Create a recipe for '{user_input}' scaled for {num_people} persons.
                2. Match these ingredients to the provided MARKET DATA.
                3. For each ingredient, find the semantic match (ensure category integrity).
                4. Include the 'package_size' from the market data in your response.
                5. Select the CHEAPEST match across all stores for the 'Hustle' route.
                6. Find the best 'One-Stop' shop options.

                MARKET DATA:
                {market_context}

                Return strictly valid JSON:
                {{
                  "recipe_name": "string",
                  "instructions": "string",
                  "hustle_basket": [
                    {{"ingredient": "item", "product": "name", "size": "package_size", "price": 0.0, "std_price": 0.0, "store": "Lidl", "discount": "20%"}}
                  ],
                  "one_stop_options": [
                    {{"store": "string", "total_price": 0.0, "coverage_pct": 0}}
                  ]
                }}
                """
                
                try:
                    resp = model.generate_content(strategy_prompt)
                    clean_json = resp.text.strip().replace("```json", "").replace("```", "")
                    strat_data = json.loads(clean_json)
                    
                    st.success(f"🍽️ **Strategy for:** {strat_data['recipe_name']} ({num_people} persons)")
                    with st.expander("👨‍🍳 View Cooking Steps"):
                        st.write(strat_data['instructions'])

                    # --- 2. DISPLAY STRATEGIC TIERS ---
                    tab_hustle, tab_one_stop = st.tabs(["🏃 Multi-Shop (Max Savings)", "🏠 One-Stop (Convenience)"])

                    with tab_hustle:
                        h_df = pd.DataFrame(strat_data['hustle_basket'])
                        hustle_total = h_df['price'].sum()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Global Minimum Cost", f"{hustle_total:.2f}€")
                        with col2:
                            total_savings = (h_df['std_price'].sum()) - hustle_total
                            st.metric("Total Realized Savings", f"{max(0, total_savings):.2f}€")

                        st.markdown("#### Your Optimized Procurement List")
                        # Now including the requested 'size' column
                        st.table(h_df[['ingredient', 'product', 'size', 'std_price', 'price', 'discount', 'store']])

                    with tab_one_stop:
                        s_df = pd.DataFrame(strat_data['one_stop_options']).sort_values(by="total_price")
                        st.table(s_df)
                        
                        premium = s_df.iloc[0]['total_price'] - hustle_total
                        st.info(f"💡 The Convenience Premium is **{premium:.2f}€**. Shopping at {s_df.iloc[0]['store']} only is { (premium/hustle_total)*100:.0f}% more expensive.")

                except Exception as e:
                    st.error(f"AI Strategy Error: {str(e)}")
else:
    st.info("Upload retailer flyers to enable the Kitchen Strategist.")
