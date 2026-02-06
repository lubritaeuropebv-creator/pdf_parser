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
            with st.spinner("Analyzing market data and creating strategy..."):
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-3-flash-preview")
                
                # 1. STANDARDIZED RECIPE GENERATION
                recipe_prompt = f"""
                Create a recipe for: '{user_input}'.
                Return strictly valid JSON with:
                - 'recipe_name': Title.
                - 'ingredients': List of generic Lithuanian grocery terms (e.g., 'Sviestas', 'Pienas', 'Kvietiniai miltai').
                - 'instructions': Short steps.
                """
                try:
                    resp = model.generate_content(recipe_prompt)
                    clean_json = resp.text.strip().replace("```json", "").replace("```", "")
                    recipe_data = json.loads(clean_json)
                    
                    st.success(f"🍽️ **Recipe:** {recipe_data['recipe_name']}")
                    with st.expander("View Cooking Instructions"):
                        st.write(recipe_data['instructions'])

                    df = st.session_state['master_df']
                    ingredients = recipe_data['ingredients']
                    
                    # --- 2. HIGH-PRECISION PROCUREMENT ENGINE ---
                    st.divider()
                    st.subheader("🛒 Procurement Strategy Optimizer")
                    
                   # --- REFINED HIGH-PRECISION MATCHING ---
                    hustle_items = []
                    missing_ingredients = []
                    df = st.session_state['master_df']
                    ingredients = recipe_data['ingredients']

                    for ing in ingredients:
                        clean_ing = ing.lower().strip()
                        
                        # STEP 1: Strict Token Matching (Prevents "Coffee vs Paprika")
                        matches = process.extract(clean_ing, df['product_name'], scorer=fuzz.token_set_ratio, limit=10)
                        valid_indices = [m[2] for m in matches if m[1] >= 80] # Adjusted to 80 for better recall
                        
                        if valid_indices:
                            candidates = df.iloc[valid_indices]
                            best = candidates.sort_values(by='disc_price').iloc[0]
                            hustle_items.append({
                                'Ingredient': ing, 
                                'Product': best['product_name'], 
                                'Price': best['disc_price'], 
                                'Store': best['store']
                            })
                        else:
                            # STEP 2: Fallback - Simple Containment Check
                            # This catches "Sausainiai" inside "Sondey Sausainiai įvairių rūšių"
                            fallback_df = df[df['product_name'].str.contains(clean_ing, case=False, na=False)]
                            
                            if not fallback_df.empty:
                                best = fallback_df.sort_values(by='disc_price').iloc[0]
                                hustle_items.append({
                                    'Ingredient': ing, 
                                    'Product': best['product_name'], 
                                    'Price': best['disc_price'], 
                                    'Store': best['store']
                                })
                            else:
                                missing_ingredients.append(ing)

                    # --- 3. ONE-STOP SHOP ANALYSIS ---
                    shop_results = []
                    for shop in df['store'].unique():
                        shop_df = df[df['store'] == shop]
                        found_count = 0
                        shop_total = 0
                        for ing in ingredients:
                            m = process.extractOne(ing, shop_df['product_name'], scorer=fuzz.token_set_ratio)
                            if m and m[1] >= 85:
                                found_count += 1
                                shop_total += shop_df[shop_df['product_name'] == m[0]]['disc_price'].min()
                        
                        if found_count > 0:
                            shop_results.append({
                                "Store": shop, "Total": round(shop_total, 2), 
                                "Coverage": f"{found_count}/{len(ingredients)}",
                                "Pct": (found_count / len(ingredients))
                            })

                    # --- 4. DISPLAY TIERS & INTERACTIVE SWAP ---
                    tab_hustle, tab_one_stop = st.tabs(["🏃 Multi-Shop (Max Savings)", "🏠 One-Stop (Convenience)"])

                    with tab_hustle:
                        if hustle_items:
                            h_df = pd.DataFrame(hustle_items)
                            hustle_total = h_df['Price'].sum()
                            st.metric("Global Minimum Cost", f"{hustle_total:.2f}€")
                            
                            for i, row in h_df.iterrows():
                                c1, c2 = st.columns([1, 2])
                                with c1:
                                    st.write(f"**{row['Ingredient']}**")
                                with c2:
                                    alts = process.extract(row['Ingredient'], df['product_name'], scorer=fuzz.token_set_ratio, limit=3)
                                    alt_opts = [f"{df.iloc[m[2]]['product_name']} ({df.iloc[m[2]]['disc_price']}€ @ {df.iloc[m[2]]['store']})" for m in alts if m[1] >= 75]
                                    if alt_opts:
                                        st.selectbox(f"Swap {row['Ingredient']}", alt_opts, key=f"h_swp_{i}", label_visibility="collapsed")
                        
                        if missing_ingredients:
                            st.warning(f"⚠️ No matches for: {', '.join(missing_ingredients)}")

                    with tab_one_stop:
                        if shop_results:
                            s_df = pd.DataFrame(shop_results).sort_values(by=["Pct", "Total"], ascending=[False, True])
                            best_one_stop = s_df.iloc[0]['Total']
                            variance = best_one_stop - hustle_total
                            
                            st.metric("One-Stop Premium", f"+{variance:.2f}€", help="Difference between single shop and multi-shop strategy")
                            st.table(s_df[["Store", "Total", "Coverage"]])

                except Exception as e:
                    st.error(f"Strategy Error: {str(e)}")
                               
else:
    st.info("Upload retailer flyers to enable the Kitchen Strategist.")
