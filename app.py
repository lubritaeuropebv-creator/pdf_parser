import streamlit as st
import json, tempfile, time, datetime
import pandas as pd
from pdf2image import convert_from_path
import google.generativeai as genai
from rapidfuzz import process, fuzz

# --- CONFIG ---
st.set_page_config(page_title="LT Multi-Shop Intelligence", layout="wide")

def extract_market_data(api_key, pdf_files):
    genai.configure(api_key=api_key)
    
    # Using Gemini 3 Flash for Agentic Vision
    model = genai.GenerativeModel(
        model_name="gemini-3-flash-preview",
        system_instruction=(
            "You are a Lithuanian retail price expert. Extract data from: LIDL, MAXIMA, IKI, RIMI, NORFA. "
            "Pay special attention to 'Loyalty Card' prices and 'Multibuy' (e.g., 2 už 1). "
            "Fields: product_name, standard_price, discounted_price, unit_price_info, discount_pct."
        )
    )
    
    all_data = []
    
    # 1. Processing all uploaded files
    pdf_pages = []
    for file, name in pdf_files:
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                images = convert_from_path(tmp.name, dpi=200)
                for i, img in enumerate(images):
                    pdf_pages.append({"name": name, "img": img, "page": i+1})
    
    if not pdf_pages:
        return pd.DataFrame()

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, item in enumerate(pdf_pages):
        status_text.write(f"🕵️‍♂️ Scanning **{item['name']}** - Page {item['page']}...")
        progress_bar.progress((idx + 1) / len(pdf_pages))
        
        prompt = f"""
        Extract every product from this {item['name']} flyer page.
        Include loyalty-card-only prices as the 'discounted_price' if applicable.
        Return a JSON list:
        [{{"product_name": "...", "standard_price": 0.0, "discounted_price": 0.0, "unit_price_info": "€/kg or €/vnt", "discount_pct": 0}}]
        """
        
        try:
            response = model.generate_content([prompt, item['img']])
            text_response = response.text.strip().replace("```json", "").replace("```", "")
            page_data = json.loads(text_response)
            for p in page_data:
                p["store"] = item["name"]
                all_data.append(p)
        except:
            continue

    return pd.DataFrame(all_data)

# --- UI ---
st.title("🛒 LT Retail Intelligence (5-Shop Comparison)")

with st.sidebar:
    st.header("1. API Authentication")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.divider()
    st.header("2. Upload Flyers")
    # Organized uploader for 5 major chains
    uploads = {
        "LIDL": st.file_uploader("Lidl PDF", type="pdf"),
        "MAXIMA": st.file_uploader("Maxima PDF", type="pdf"),
        "IKI": st.file_uploader("IKI PDF", type="pdf"),
        "RIMI": st.file_uploader("Rimi PDF", type="pdf"),
        "NORFA": st.file_uploader("Norfa PDF", type="pdf")
    }
    
    go = st.button("🚀 Analyze All Markets", use_container_width=True)

if go and api_key:
    active_uploads = [(f, n) for n, f in uploads.items() if f is not None]
    if active_uploads:
        df = extract_market_data(api_key, active_uploads)
        if not df.empty:
            st.session_state['df'] = df
            st.success(f"Strategic Map Complete: {len(df)} products analyzed.")

# --- STRATEGIC ANALYSIS DASHBOARD ---
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Offers", len(df))
    c2.metric("Shops Analyzed", df['store'].nunique())
    c3.metric("Avg. Discount", f"{int(df['discount_pct'].mean())}%")

    st.divider()
    
    # Multi-Shop Comparison Filter
    search = st.text_input("Search for a specific product across all 5 shops:", "Pienas")
    if search:
        # Using fuzzy matching to find the item across different shop naming conventions
        matches = process.extract(search, df['product_name'].tolist(), scorer=fuzz.WRatio, limit=10)
        found_names = [m[0] for m in matches if m[1] > 60]
        results = df[df['product_name'].isin(found_names)].sort_values(by="discounted_price")
        
        st.write(f"### Best Prices for '{search}'")
        st.dataframe(results[['product_name', 'store', 'discounted_price', 'unit_price_info', 'discount_pct']], use_container_width=True)

    with st.expander("📂 View Full Database"):
        st.dataframe(df, use_container_width=True)
