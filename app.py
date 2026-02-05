import streamlit as st
import pandas as pd
import plotly.express as px
import json, gc, io, itertools
from pdf2image import convert_from_bytes
import google.generativeai as genai
import fitz  # PyMuPDF
from rapidfuzz import process, fuzz

# --- 1. CONFIGURATION ---
# Forces sidebar to be visible and sets wide layout for the dashboard
st.set_page_config(
    page_title="LT Retail Strategist Pro", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Initialize global storage
if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()

# --- 2. PROCESSING ENGINE (MEMORY SAFE) ---
def process_flyer(api_key, uploaded_file, shop_name):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")
    extracted_data = []
    
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    progress = st.progress(0)
    status = st.empty()

    for i in range(len(doc)):
        status.text(f"Scanning {shop_name}: Page {i+1}/{len(doc)}")
        # Convert only 1 page to save RAM
        images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1, dpi=150)
        
        if images:
            prompt = "Extract product data JSON: name, std_price, disc_price, unit_price, disc_pct. Lithuanian names."
            try:
                response = model.generate_content([prompt, images[0]])
                clean = response.text.strip().replace("```json", "").replace("```", "")
                page_items = json.loads(clean)
                for item in page_items:
                    item["store"] = shop_name
                    extracted_data.append(item)
            except:
                continue
            del images
            gc.collect()
        progress.progress((i + 1) / len(doc))
    
    doc.close()
    return pd.DataFrame(extracted_data)

# --- 3. AI RECIPE & OPTIMIZER LOGIC ---
def get_recipe(api_key, idea):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")
    prompt = f"Idea: {idea}. Return JSON: {{'recipe_name': '...', 'ingredients': ['lt_name1', 'lt_name2'], 'steps': '...'}}"
    try:
        res = model.generate_content(prompt)
        return json.loads(res.text.strip().replace("```json", "").replace("```", ""))
    except: return None

def find_best_cart(ingredients, df, max_shops):
    stores = df['store'].unique()
    best_total = float('inf')
    best_plan = None
    
    for combo in itertools.combinations(stores, max_shops):
        subset = df[df['store'].isin(combo)]
        current_total = 0
        current_items = []
        found_count = 0

        for ing in ingredients:
            choices = subset['product_name'].tolist()
            match = process.extractOne(ing, choices, scorer=fuzz.WRatio)
            if match and match[1] > 65:
                row = subset[subset['product_name'] == match[0]].iloc[0]
                current_total += row['disc_price']
                current_items.append({"Item": ing, "Match": row['product_name'], "Store": row['store'], "Price": row['disc_price']})
                found_count += 1
        
        if current_total < best_total and found_count > 0:
            best_total = current_total
            best_plan = (combo, current_items, current_total)
    
    return best_plan

# --- 4. SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.header("🔑 Authentication")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.divider()
    st.header("📄 Flyer Upload")
    shop_name = st.selectbox("Select Retailer", ["Lidl", "Maxima", "IKI", "Rimi", "Norfa"])
    file = st.file_uploader(f"Upload {shop_name} PDF", type="pdf")
    
    if st.button("🚀 Process & Add to Database"):
        if api_key and file:
            new_data = process_flyer(api_key, file, shop_name)
            st.session_state['master_df'] = pd.concat([st.session_state['master_df'], new_data], ignore_index=True).drop_duplicates()
            st.success(f"Added {len(new_data)} items!")

# --- 5. MAIN DASHBOARD ---
st.title("🛒 LT Retail Strategy & AI Chef")

if not st.session_state['master_df'].empty:
    df = st.session_state['master_df']
    
    # --- TAB 1: ANALYTICS ---
    tab1, tab2 = st.tabs(["📊 Price Analytics", "🍳 AI Recipe Optimizer"])
    
    with tab1:
        st.subheader("Market Comparison")
        item_to_check = st.selectbox("Compare Price of:", sorted(df['product_name'].unique()))
        compare_df = df[df['product_name'] == item_to_check]
        fig = px.bar(compare_df, x='store', y='disc_price', color='store', text='disc_price')
        st.plotly_chart(fig, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Master CSV", csv, "all_shops.csv", "text/csv")

    # --- TAB 2: RECIPE & CART ---
    with tab2:
        idea = st.text_input("I want to cook...", "Balandėliai")
        shops_allowed = st.slider("Max stores to visit", 1, 5, 2)
        
        if st.button("🪄 Plan My Meal"):
            recipe = get_recipe(api_key, idea)
            if recipe:
                st.subheader(recipe['recipe_name'])
                plan = find_best_cart(recipe['ingredients'], df, shops_allowed)
                if plan:
                    combo, items, total = plan
                    st.success(f"Cheapest Total: {total:.2f}€")
                    st.write(f"**Go to:** {' & '.join(combo)}")
                    st.table(pd.DataFrame(items))
                    
                    missing = [i for i in recipe['ingredients'] if i not in [x['Item'] for x in items]]
                    if missing: st.warning(f"⚠️ Pay full price for: {', '.join(missing)}")
                else: st.error("No items found in flyers.")

else:
    st.info("👈 Please enter your API Key and upload a PDF in the sidebar to begin.")
