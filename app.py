import streamlit as st
import pandas as pd
import json, gc, re, time
from pdf2image import convert_from_bytes
import google.generativeai as genai
import fitz 
from rapidfuzz import process, fuzz

# --- 1. CONFIG & SESSION STATE ---
st.set_page_config(
    page_title="LT AI Kitchen Strategist", 
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'master_df' not in st.session_state:
    st.session_state['master_df'] = pd.DataFrame()
if 'last_batch_df' not in st.session_state:
    st.session_state['last_batch_df'] = pd.DataFrame()
if 'last_shop_name' not in st.session_state:
    st.session_state['last_shop_name'] = ""

# --- 2. SEMANTIC & MATH UTILITIES ---
def get_base_word(word):
    """Strips Lithuanian inflections to improve matching (e.g., Vištienos -> Vištiena)."""
    word = word.lower().strip()
    suffixes = ['ių', 'os', 'as', 'us', 'is', 'ą', 'ę', 'į', 'ų', 'oje', 'io', 'iai']
    for s in suffixes:
        if word.endswith(s) and len(word) > 3:
            return word[:-len(s)]
    return word

def normalize_lt_price(value):
    if value is None or value == "": return 0.0
    s = str(value).lower().replace(',', '.').replace('€', '').strip()
    try:
        clean = re.findall(r"\d+\.\d+|\d+", s)
        return float(clean[0]) if clean else 0.0
    except: return 0.0

def parse_weight(size_str):
    """Normalizes weight to KG/L and fixes the x1000 parsing error."""
    if not size_str: return 0.0
    s = str(size_str).lower().replace(',', '.').strip()
    try:
        multiplier = 1.0
        if 'x' in s:
            parts = s.split('x')
            multiplier = float(re.findall(r"\d+\.?\d*", parts[0])[0])
            s = parts[1]

        nums = re.findall(r"\d+\.?\d*", s)
        if not nums: return 0.0
        val = float(nums[0]) * multiplier

        if 'mg' in s: return val / 1_000_000
        if 'kg' in s or ('l' in s and 'ml' not in s): return val
        if 'g' in s or 'ml' in s: return val / 1000 
        return val / 1000 if val > 20 else val
    except: return 0.0

def get_candidate_matches(ingredient, df, limit=3):
    """Finds top candidates for swapping items."""
    if df.empty: return []
    choices = df['product_name'].tolist()
    results = process.extract(ingredient, choices, scorer=fuzz.token_set_ratio, limit=10)
    
    candidates = []
    seen = set()
    for match_text, score, idx in results:
        if score < 45: continue
        prod = df.iloc[idx]
        if prod['product_name'] not in seen:
            candidates.append({
                "Product": prod['product_name'],
                "Price": prod['disc_price'],
                "Store": prod['store'],
                "Size": prod['package_size'],
                "Score": score
            })
            seen.add(prod['product_name'])
    return sorted(candidates, key=lambda x: (-x['Score'], x['Price']))[:limit]

# --- 3. SIDEBAR: DATA INGESTION ---
with st.sidebar:
    st.header("🔑 Authenticate")
    api_key = st.text_input("Gemini API Key", type="password")
    
    if not st.session_state['master_df'].empty:
        st.divider()
        st.subheader("💾 Data Hub")
        csv_master = st.session_state['master_df'].to_csv(index=False).encode('utf-8')
        st.download_button("Download Master DB", csv_master, "LT_Master_Prices.csv", "text/csv", type="primary")

    st.divider()
    st.header("📄 Upload Flyer")
    shop = st.selectbox("Retailer", ["Lidl", "Maxima", "IKI", "Rimi", "Norfa"])
    file = st.file_uploader(f"Upload {shop} PDF", type="pdf")
    
    if st.button("🚀 Deep Scan & Audit", width="stretch"):
        if api_key and file:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            extracted = []
            
            progress_bar = st.progress(0)
            for i in range(len(doc)):
                # High DPI (220) for small text detection
                images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1, dpi=220)
                prompt = "Extract products. JSON: 'product_name', 'package_size', 'std_price', 'disc_price'."
                try:
                    resp = model.generate_content([prompt, images[0]])
                    items = json.loads(resp.text.strip().replace("```json", "").replace("```", ""))
                    for it in items:
                        std = normalize_lt_price(it.get('std_price'))
                        dp = normalize_lt_price(it.get('disc_price'))
                        pkg = str(it.get('package_size', ''))
                        w = parse_weight(pkg)
                        extracted.append({
                            "product_name": it.get('product_name', 'Unknown'),
                            "package_size": pkg,
                            "std_price": std,
                            "disc_price": dp if dp > 0 else std,
                            "unit_price_kg": round(dp / w, 2) if w > 0 else 0.0,
                            "discount_pct": round(((std - dp) / std * 100), 1) if std > dp > 0 else 0,
                            "store": shop
                        })
                except: pass
                progress_bar.progress((i + 1) / len(doc))
            
            new_df = pd.DataFrame(extracted)
            st.session_state['last_batch_df'] = new_df
            st.session_state['last_shop_name'] = shop
            st.session_state['master_df'] = pd.concat([st.session_state['master_df'], new_df], ignore_index=True).drop_duplicates()
            st.rerun()

# --- 4. MAIN DASHBOARD ---
st.title("🛡️ LT AI Kitchen Strategist")

if not st.session_state['last_batch_df'].empty:
    with st.expander(f"✅ {st.session_state['last_shop_name']} Processed - Save CSV", expanded=True):
        csv_batch = st.session_state['last_batch_df'].to_csv(index=False).encode('utf-8')
        st.download_button("Download Latest Batch", csv_batch, f"{st.session_state['last_shop_name']}_data.csv")

if not st.session_state['master_df'].empty:
    tab1, tab2 = st.tabs(["📊 Market Data", "👨‍🍳 AI Recipe & Swap"])
    
    with tab1:
        st.dataframe(st.session_state['master_df'], use_container_width=True)

    with tab2:
        user_input = st.text_input("Meal Idea or Ingredients:", placeholder="e.g., 'Koldūnai su grybais'")
        if st.button("Generate & Optimize"):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"Recipe for {user_input}. Return JSON: 'name', 'ingredients' (List simple LT names), 'steps'."
            recipe = json.loads(model.generate_content(prompt).text.strip().replace("```json", "").replace("```", ""))
            
            st.markdown(f"### 🍽️ {recipe['name']}")
            basket = []
            for ing in recipe['ingredients']:
                candidates = get_candidate_matches(ing, st.session_state['master_df'])
                with st.expander(f"Item: {ing}", expanded=True):
                    if candidates:
                        best = candidates[0]
                        st.write(f"**Selected:** {best['Product']} at **{best['Store']}** ({best['Price']}€)")
                        if len(candidates) > 1:
                            st.markdown("*Alternative Swaps:*")
                            for alt in candidates[1:]:
                                st.write(f"- {alt['Product']} ({alt['Store']}): {alt['Price']}€")
                        basket.append(best)
                    else: st.warning("No flyer match found.")
            
            if basket:
                st.divider()
                st.metric("Total Optimal Price", f"{sum(i['Price'] for i in basket):.2f} €")
else:
    st.info("Upload flyer PDFs to start.")
