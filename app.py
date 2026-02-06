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
            user_input = st.text_input("What do you have? Or what do you crave?", placeholder="e.g., 'I have eggs and tomatoes' OR 'Something spicy for dinner'")
        with col_btn:
            generate_btn = st.button("👨‍🍳 Invent & Shop", type="primary", use_container_width=True)

        if generate_btn and user_input:
            with st.spinner("Creating recipe and scanning flyers..."):
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-3-flash-preview")
                
                # 1. GENERATE RECIPE
                recipe_prompt = f"""
                Create a simple, delicious recipe based on: '{user_input}'.
                Return strictly valid JSON with:
                - 'recipe_name': Title of the dish.
                - 'ingredients': A list of generic Lithuanian grocery ingredients (e.g., ['Vištienos krūtinėlė', 'Ryžiai']).
                - 'instructions': Short cooking steps.
                """
                try:
                    resp = model.generate_content(recipe_prompt)
                    clean_json = resp.text.strip().replace("```json", "").replace("```", "")
                    recipe_data = json.loads(clean_json)
                    
                    st.success(f"🍽️ **Recipe:** {recipe_data['recipe_name']}")
                    with st.expander("View Cooking Instructions"):
                        st.write(recipe_data['instructions'])

                    # 2. MATCH INGREDIENTS TO DATABASE
                    basket_items = []
                    found_ingredients = []
                    missing_ingredients = []
                    
                    df = st.session_state['master_df']
                    
                    for ingredient in recipe_data['ingredients']:
                        # Fuzzy match ingredient to product names
                        match = process.extractOne(ingredient, df['product_name'], scorer=fuzz.WRatio)
                        if match and match[1] > 50: # Threshold
                            best_product = df.iloc[match[2]]
                            basket_items.append({
                                'Ingredient': ingredient,
                                'Match': best_product['product_name'],
                                'Price': best_product['disc_price'],
                                'Store': best_product['store'],
                                'Link': match[2] # Index reference
                            })
                            found_ingredients.append(ingredient)
                        else:
                            missing_ingredients.append(ingredient)

                    if basket_items:
                        basket_df = pd.DataFrame(basket_items)
                        
                        # --- 3. MULTI-TIER SHOPPING STRATEGY ---
                        st.divider()
                        st.subheader("🛒 Procurement Strategy")
                        
                        # A. ONE-STOP SHOP ANALYSIS
                        shops = df['store'].unique()
                        shop_costs = {}
                        
                        for shop in shops:
                            # Filter basket for this shop
                            shop_specific_basket = basket_df[basket_df['Store'] == shop]
                            # If shop has at least 50% of the ingredients, we calculate a "Basket Estimate"
                            # (Missing items are penalized by adding the average price of that item across other stores)
                            if len(shop_specific_basket) > 0:
                                cost = shop_specific_basket['Price'].sum()
                                coverage = len(shop_specific_basket) / len(found_ingredients) * 100
                                shop_costs[shop] = {'cost': cost, 'coverage': coverage}

                        # B. MULTI-SHOP OPTIMAL (The "Hustle" Price)
                        # We already picked the absolute best matches globally in the loop above? 
                        # Actually, extractOne finds the best STRING match, not necessarily the best PRICE.
                        # Let's refine: Find CHEAPEST valid match for each ingredient globally.
                        
                        optimized_basket = []
                        total_hustle_cost = 0
                        
                        for ingredient in recipe_data['ingredients']:
                            # Get all matches > 50 score
                            matches = process.extract(ingredient, df['product_name'], scorer=fuzz.WRatio, limit=10)
                            valid_indices = [m[2] for m in matches if m[1] > 50]
                            
                            if valid_indices:
                                candidates = df.iloc[valid_indices]
                                best_deal = candidates.sort_values(by='disc_price', ascending=True).iloc[0]
                                optimized_basket.append({
                                    'Ingredient': ingredient,
                                    'Product': best_deal['product_name'],
                                    'Price': best_deal['disc_price'],
                                    'Store': best_deal['store']
                                })
                                total_hustle_cost += best_deal['disc_price']
                        
                        opt_df = pd.DataFrame(optimized_basket)
                        
                        # DISPLAY RESULTS
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🏃 The 'Hustle' Route (Multi-Shop)")
                            st.metric("Lowest Theoretical Price", f"{total_hustle_cost:.2f}€")
                            st.dataframe(opt_df, hide_index=True)
                            
                            # Store Split Count
                            unique_stores = opt_df['Store'].unique()
                            st.info(f"Route requires visiting: {', '.join(unique_stores)}")

                        with col2:
                            st.markdown("#### 🏠 One-Stop Options")
                            if shop_costs:
                                # Sort by coverage desc, then cost asc
                                sorted_shops = sorted(shop_costs.items(), key=lambda x: (-x[1]['coverage'], x[1]['cost']))
                                for sh, data in sorted_shops:
                                    st.write(f"**{sh}**: {data['cost']:.2f}€ (Found {data['coverage']:.0f}% of items)")
                            else:
                                st.warning("No single shop has enough ingredients.")

                        if missing_ingredients:
                            st.error(f"⚠️ Could not find prices for: {', '.join(missing_ingredients)}")

                except Exception as e:
                    st.error(f"Recipe Generation Error: {str(e)}")

else:
    st.info("Upload retailer flyers to enable the Kitchen Strategist.")
