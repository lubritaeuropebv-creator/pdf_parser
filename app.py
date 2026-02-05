import streamlit as st
import json, tempfile, time, datetime, re
import pandas as pd
from pdf2image import convert_from_path
import google.generativeai as genai

# --- CONFIG ---
st.set_page_config(page_title="LT Price Strategist Pro", layout="wide")

def extract_strategic_data(api_key, pdf_files):
    genai.configure(api_key=api_key)
    
    # Using Gemini 3 Flash for PhD-level reasoning at speed
    model = genai.GenerativeModel(
        model_name="gemini-3-flash-preview",
        system_instruction=(
            "You are an elite retail auditor. Your goal is to extract every detail from the price tags. "
            "Search for small text indicating unit price (e.g., '€/kg', '€/vnt'). "
            "Calculate percentage discounts if not explicitly stated."
        )
    )
    
    all_data = []
    
    # PDF to Image conversion
    pdf_pages = []
    for file, name in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            # 200 DPI is required to read the tiny 'Price per kg' text accurately
            images = convert_from_path(tmp.name, dpi=200)
            for i, img in enumerate(images):
                pdf_pages.append({"name": name, "img": img, "page": i+1})
    
    progress_bar = st.progress(0)
    
    for idx, item in enumerate(pdf_pages):
        progress_bar.progress((idx + 1) / len(pdf_pages))
        
        # AGGRESSIVE PROMPT for full metrics
        prompt = """
        Analyze this flyer page. For EVERY product, return a JSON list:
        - product_name: Full name including brand and size (e.g., 'Dvaro pienas 1L')
        - standard_price: Regular price before discount (float)
        - discounted_price: Sale price (float)
        - unit_price_info: Price per kg, liter, or piece (string, e.g., '2.49 €/kg')
        - discount_pct: Percent off (integer, e.g., 30)
        
        Rules: 
        1. If discount_pct is missing, calculate it: ((std - disc) / std) * 100.
        2. Look for unit prices in the very small text at the bottom of labels.
        3. ONLY return the JSON list.
        """
        
        try:
            response = model.generate_content([prompt, item['img']])
            text_response = response.text.strip()
            
            # JSON Cleaning
            if "```json" in text_response:
                text_response = text_response.split("```json")[1].split("```")[0]
            
            page_data = json.loads(text_response)
            for p in page_data:
                p["store"] = item["name"]
                p["page"] = item["page"]
                all_data.append(p)
        except:
            continue

    return pd.DataFrame(all_data)

# --- UI ---
st.title("🛒 LT Strategic Price Intelligence")

with st.sidebar:
    st.header("Upload Flyers")
    api_key = st.text_input("Gemini API Key", type="password")
    lidl = st.file_uploader("Lidl PDF", type="pdf")
    maxima = st.file_uploader("Maxima PDF", type="pdf")
    go = st.button("🚀 Analyze Market Prices")

if go and api_key:
    df = extract_strategic_data(api_key, [(lidl, "Lidl"), (maxima, "Maxima")])
    if not df.empty:
        st.session_state['df'] = df
        st.success(f"Successfully tracked {len(df)} products!")

# --- DISPLAY & STRATEGY ---
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # Strategist's View
    st.subheader("📊 Market Analysis Table")
    
    # Add a CSS hack to highlight the best discounts
    st.dataframe(
        df.sort_values(by="discount_pct", ascending=False),
        column_config={
            "discount_pct": st.column_config.ProgressColumn("Discount %", min_value=0, max_value=100, format="%d%%"),
            "discounted_price": st.column_config.NumberColumn("Sale Price", format="%.2f €"),
            "standard_price": st.column_config.NumberColumn("Std Price", format="%.2f €"),
        },
        use_container_width=True
    )

    # Negotiator Highlight
    top_deal = df.loc[df['discount_pct'].idxmax()]
    st.warning(f"💡 **Negotiator Insight:** The best value-to-cost deal is **{top_deal['product_name']}** at **{top_deal['store']}** ({top_deal['discount_pct']}% off).")
