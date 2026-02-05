import streamlit as st
import json, os, datetime, itertools, tempfile
import pandas as pd
from pdf2image import convert_from_path
import google.generativeai as genai
from rapidfuzz import process, fuzz
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="LT Promo Intelligence", layout="wide")

# --- 1. CORE LOGIC FUNCTIONS ---

def find_best_match(target_item, promo_df, threshold=70):
    """Handles Lithuanian declensions (pienas/pieno) using fuzzy logic."""
    choices = promo_df["product_name"].tolist()
    best_match = process.extractOne(target_item, choices, scorer=fuzz.WRatio)
    if best_match and best_match[1] >= threshold:
        return promo_df[promo_df["product_name"] == best_match[0]].iloc[0]
    return None

def get_savings_plan(cart, df):
    """Calculates the best 2-shop combo and potential savings."""
    stores = df["store"].unique()
    best_combo, best_total_disc, best_total_std, best_assignment = None, float("inf"), 0, []

    for s1, s2 in itertools.combinations(stores, 2):
        subset = df[df["store"].isin([s1, s2])]
        curr_assign, curr_disc, curr_std = [], 0, 0
        
        for item in cart:
            match = find_best_match(item, subset)
            if match is not None:
                std_p = match["standard_price"] if pd.notna(match["standard_price"]) else round(match["discount_price"] / 0.7, 2)
                curr_assign.append({
                    "Item": item, "Found": match["product_name"], 
                    "Store": match["store"], "Price": match["discount_price"], 
                    "Standard": std_p, "Savings": round(std_p - match["discount_price"], 2)
                })
                curr_disc += match["discount_price"]
                curr_std += std_p

        if curr_disc < best_total_disc and curr_assign:
            best_total_disc, best_total_std, best_combo, best_assignment = curr_disc, curr_std, (s1, s2), curr_assign

    return best_combo, round(best_total_disc, 2), round(best_total_std, 2), pd.DataFrame(best_assignment)

# --- 2. STREAMLIT UI ---

st.title("🛒 Lithuania Promo Intelligence")
st.markdown("Automated Price Strategy & Flyer Analysis")

with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password")
    lidl_pdf = st.file_uploader("Lidl Flyer", type="pdf")
    maxima_pdf = st.file_uploader("Maxima Flyer", type="pdf")
    process_btn = st.button("Analyze Flyers")

if process_btn:
    if not api_key or not lidl_pdf or not maxima_pdf:
        st.error("Please provide API Key and both PDF files.")
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        all_data = []
        for file, name in [(lidl_pdf, "LIDL"), (maxima_pdf, "MAXIMA")]:
            with st.status(f"Processing {name}...") as status:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getvalue())
                    pages = convert_from_path(tmp.name, dpi=150)
                
                for i, page in enumerate(pages):
                    prompt = "Extract retail promos into JSON: product_name, category, standard_price, discount_price, discount_percent."
                    response = model.generate_content([prompt, page])
                    try:
                        clean_json = response.text.replace("```json", "").replace("```", "")
                        items = json.loads(clean_json)
                        for item in items:
                            item["store"] = name
                            all_data.append(item)
                    except: continue
                status.update(label=f"{name} Complete!", state="complete")
        
        st.session_state['df'] = pd.DataFrame(all_data)

# --- 3. RESULTS DASHBOARD ---

if 'df' in st.session_state:
    df = st.session_state['df']
    cart_input = st.text_input("Your Shopping List (comma separated)", "pienas, sūris, vištiena, obuoliai")
    
    if st.button("Optimize My Route"):
        cart = [x.strip() for x in cart_input.split(",")]
        combo, total, std, plan_df = get_savings_plan(cart, df)
        
        if not plan_df.empty:
            savings = round(std - total, 2)
            
            # Key Performance Indicators
            col1, col2, col3 = st.columns(3)
            col1.metric("Optimal Cost", f"{total} €")
            col2.metric("Total Savings", f"{savings} €")
            col3.metric("Combo", f"{combo[0]} + {combo[1]}")
            
            st.table(plan_df)
            
            # Strategy Note
            savings_pct = (savings / std) * 100
            st.info(f"💡 Strategy: You are saving **{savings_pct:.1f}%** by splitting your trip.")
