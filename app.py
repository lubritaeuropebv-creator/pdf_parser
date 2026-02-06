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
            with st.spinner("Creating recipe and scanning flyers..."):
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-3-flash-preview")
                
                # 1. OPTIMIZED RECIPE GENERATION
                # We force the AI to use standard Lithuanian terms to improve matching accuracy.
                recipe_prompt = f"""
                Create a simple recipe based on: '{user_input}'.
                Return strictly valid JSON with:
                - 'recipe_name': Title.
                - 'ingredients': List of generic Lithuanian grocery terms (e.g., 'Sviestas', 'Pienas', 'Kiaušiniai').
                - 'instructions': Short steps.
                """
                try:
                    resp = model.generate_content(recipe_prompt)
                    clean_json = resp.text.strip().replace("```json", "").replace("```", "")
                    recipe_data = json.loads(clean_json)
                    
                    st.success(f"🍽️ **Recipe:** {recipe_data['recipe_name']}")
                    with st.expander("View Cooking Instructions"):
                        st.write(recipe_data['instructions'])

                    # 2. REFINED MATCHING LOGIC
                    optimized_basket = []
                    missing_ingredients = []
                    df = st.session_state['master_df']
                    
                    for ingredient in recipe_data['ingredients']:
                        # Increase threshold to 80% to avoid "Sausainiai/Ciabatta" style mismatches
                        matches = process.extract(ingredient, df['product_name'], scorer=fuzz.WRatio, limit=5)
                        valid_matches = [m for m in matches if m[1] >= 80]
                        
                        if valid_matches:
                            # From the high-confidence matches, pick the one with the lowest price
                            valid_indices = [m[2] for m in valid_matches]
                            candidates = df.iloc[valid_indices]
                            best_deal = candidates.sort_values(by='disc_price', ascending=True).iloc[0]
                            
                            optimized_basket.append({
                                'Ingredient': ingredient,
                                'Product': best_deal['product_name'],
                                'Price': best_deal['disc_price'],
                                'Store': best_deal['store']
                            })
                        else:
                            missing_ingredients.append(ingredient)

                    # 3. MULTI-TIER PROCUREMENT STRATEGY
                    if optimized_basket:
                        st.divider()
                        st.subheader("🛒 Smart Cart Optimizer")
                        
                        # Data setup for analysis
                        opt_df = pd.DataFrame(optimized_basket)
                        all_shops = df['store'].unique()
                        total_ingredients = len(recipe_data['ingredients'])
                        
                        # --- CALCULATION ENGINE ---
                        shop_analysis = []
                        for shop in all_shops:
                            # Filter the master DB for this shop and match against recipe ingredients
                            shop_items = []
                            shop_total = 0
                            found_count = 0
                            
                            for ing in recipe_data['ingredients']:
                                # Higher threshold (80) to ensure quality matches per shop
                                m = process.extractOne(ing, df[df['store'] == shop]['product_name'], scorer=fuzz.WRatio)
                                if m and m[1] >= 80:
                                    price = df.iloc[m[2]]['disc_price']
                                    shop_total += price
                                    found_count += 1
                                    shop_items.append(ing)
                            
                            coverage = (found_count / total_ingredients) * 100
                            if found_count > 0:
                                shop_analysis.append({
                                    "Store": shop,
                                    "Total Price": round(shop_total, 2),
                                    "Coverage": f"{found_count}/{total_ingredients} ({coverage:.0f}%)",
                                    "Missing": total_ingredients - found_count,
                                    "Score": coverage - (shop_total / 10) # Simple heuristic: high coverage, low price
                                })

                        # --- DISPLAY TIERS ---
                        t1, t2 = st.tabs(["📊 One-Stop Shop (Convenience)", "🏃 Multi-Shop (Max Savings)"])

                        with t1:
                            st.markdown("#### Best Single-Store Options")
                            if shop_analysis:
                                analysis_df = pd.DataFrame(shop_analysis).sort_values(by="Score", ascending=False)
                                st.table(analysis_df[["Store", "Total Price", "Coverage", "Missing"]])
                                st.info("💡 Pro Tip: The 'Best' shop balances price with how many items they actually have in stock.")
                            else:
                                st.warning("No single shop has a high-confidence match for these items.")

                        with t2:
                            col_a, col_b = st.columns([2, 1])
                            with col_a:
                                st.markdown("#### The 'Hustle' Route (Lowest Price Globally)")
                                st.dataframe(opt_df[['Ingredient', 'Product', 'Price', 'Store']], hide_index=True)
                            
                            with col_b:
                                total_hustle = opt_df['Price'].sum()
                                st.metric("Global Minimum", f"{total_hustle:.2f}€")
                                st.write("**Store Split:**")
                                st.write(opt_df['Store'].value_counts())

                        # --- SWAP FEATURE (Embedded for the 'Hustle' route) ---
                        with st.expander("🔄 Manually Swap/Refine Hustle Items"):
                            final_hustle = []
                            for i, item in enumerate(optimized_basket):
                                alt_matches = process.extract(item['Ingredient'], df['product_name'], scorer=fuzz.WRatio, limit=3)
                                alt_indices = [m[2] for m in alt_matches if m[1] >= 80]
                                alt_df = df.iloc[alt_indices]
                                options = [f"{r['product_name']} | {r['disc_price']}€ ({r['store']})" for _, r in alt_df.iterrows()]
                                
                                if options:
                                    sel = st.selectbox(f"Refine: {item['Ingredient']}", options, key=f"swp_{i}")
                                    # Logic to capture refined price for a final summary could go here

                except Exception as e:
                    st.error(f"Logic Error: {str(e)}")

                  
else:
    st.info("Upload retailer flyers to enable the Kitchen Strategist.")
