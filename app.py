import streamlit as st
import json, os, datetime, itertools, tempfile, time
import pandas as pd
from pdf2image import convert_from_path
import google.generativeai as genai
from rapidfuzz import process, fuzz
import fitz  # PyMuPDF
from io import BytesIO

# ============================================
# 1. INITIAL CONFIGURATION
# ============================================
st.set_page_config(page_title="LT Promo Intelligence Pro", layout="wide")

if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ============================================
# 2. CORE INTELLIGENCE FUNCTIONS
# ============================================

def is_promo_page(model, image, page_text=""):
    """Determines if a page should be skipped or processed."""
    # Layer 1: Local Keyword Check
    keywords = ["€", "akcija", "nuolaida", "kaina", "nuo", "tik", "proc"]
    if any(k in page_text.lower() for k in keywords):
        return True, "Keywords found"

    # Layer 2: Visual Triage (Low-res check)
    thumb = image.copy()
    thumb.thumbnail((400, 400))
    triage_prompt = "Is this a retail flyer page with multiple products and prices? Answer only 'YES' or 'NO'."
    try:
        response = model.generate_content([triage_prompt, thumb])
        if "YES" in response.text.upper():
            return True, "AI Triage Confirmed"
    except:
        return True, "Error - processing anyway"
    return False, "Non-promo/Cover page"

def find_best_match(target_item, promo_df, threshold=70):
    """Handles Lithuanian grammar (pienas/pieno) using fuzzy stems."""
    choices = promo_df["product_name"].tolist()
    best_match = process.extractOne(target_item, choices, scorer=fuzz.WRatio)
    if best_match and best_match[1] >= threshold:
        return promo_df[promo_df["product_name"] == best_match[0]].iloc[0]
    return None

def get_savings_plan(cart, df):
    """Calculates optimal 2-shop route and ROI."""
    stores = df["store"].unique()
    best_combo, best_total_disc, best_total_std, best_assignment = None, float("inf"), 0, []

    for s1, s2 in itertools.combinations(stores, 2):
        subset = df[df["store"].isin([s1, s2])]
        curr_assign, curr_disc, curr_std = [], 0, 0
        
        for item in cart:
            match = find_best_match(item, subset)
            if match is not None:
                std_p = match["standard_price"] if pd.notna(match.get("standard_price")) else round(match["discount_price"] / 0.7, 2)
                curr_assign.append({
                    "Item": item, "Found": match["product_name"], "Store": match["store"],
                    "Price": match["discount_price"], "Standard": std_p, "Savings": round(std_p - match["discount_price"], 2)
                })
                curr_disc += match["discount_price"]
                curr_std += std_p

        if curr_disc < best_total_disc and curr_assign:
            best_total_disc, best_total_std, best_combo, best_assignment = curr_disc, curr_std, (s1, s2), curr_assign

    return best_combo, round(best_total_disc, 2), round(best_total_std, 2), pd.DataFrame(best_assignment)

# ============================================
# 3. UI - SIDEBAR & UPLOAD
# ============================================

st.title("🛒 Promo Intelligence System (Lithuania)")
st.markdown("Automated Flyer Extraction • Declension Matching • Strategy Engine")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    st.divider()
    lidl_pdf = st.file_uploader("Upload Lidl Flyer", type="pdf")
    maxima_pdf = st.file_uploader("Upload Maxima Flyer", type="pdf")
    
    process_btn = st.button("🚀 Run Analysis", use_container_width=True)

# ============================================
# 4. PROCESSING ENGINE (with ETR & Smart Skip)
# ============================================

if process_btn:
    if not api_key or not lidl_pdf or not maxima_pdf:
        st.error("Missing API Key or PDF files.")
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        all_data = []
        files_to_process = [(lidl_pdf, "LIDL"), (maxima_pdf, "MAXIMA")]
        
        overall_progress = st.progress(0)
        status_text = st.empty()
        
        # Pre-count pages
        pdf_pages_map = {}
        total_total_pages = 0
        skipped_count = 0
        
        for file, name in files_to_process:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                pages = convert_from_path(tmp.name, dpi=130)
                pdf_pages_map[name] = (pages, fitz.open(stream=file.getvalue(), filetype="pdf"))
                total_total_pages += len(pages)

        start_time = time.time()
        pages_done = 0

        for name, (pages, doc) in pdf_pages_map.items():
            for i, page_img in enumerate(pages):
                pages_done += 1
                
                # Update ETR
                elapsed = time.time() - start_time
                avg_time = elapsed / pages_done
                etr = int(avg_time * (total_total_pages - pages_done))
                
                status_text.markdown(f"**Analyzing {name}**: Page {i+1}/{len(pages)} | ⏳ ETR: {etr}s")
                overall_progress.progress(pages_done / total_total_pages)

                # Smart Skip Check
                local_text = doc[i].get_text()
                should_process, reason = is_promo_page(model, page_img, local_text)
                
                if not should_process:
                    skipped_count += 1
                    continue

                # Full Extraction
                prompt = """Extract ALL promo products into a JSON list. 
                Fields: product_name, category, standard_price (float), discount_price (float). 
                Return ONLY JSON."""
                try:
                    response = model.generate_content([prompt, page_img])
                    clean_json = response.text.strip().replace("```json", "").replace("```", "")
                    items = json.loads(clean_json)
                    for item in items:
                        item["store"] = name
                        all_data.append(item)
                except:
                    continue

        status_text.success(f"Done! Processed {total_total_pages} pages ({skipped_count} skipped) in {int(time.time()-start_time)}s.")
        st.session_state['df'] = pd.DataFrame(all_data)

# ============================================
# 5. RESULTS DASHBOARD
# ============================================

if not st.session_state['df'].empty:
    df = st.session_state['df']
    
    st.divider()
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.subheader("📝 Shopping List")
        cart_input = st.text_area("Enter items (one per line or comma separated)", "pienas, sūris, vištiena, obuoliai")
        cart = [x.strip() for x in cart_input.replace("\n", ",").split(",") if x.strip()]
        run_plan = st.button("Calculate Best Strategy")

    if run_plan:
        combo, total, std, plan_df = get_savings_plan(cart, df)
        
        if not plan_df.empty:
            savings = round(std - total, 2)
            savings_pct = round((savings / std) * 100) if std > 0 else 0
            
            with col_r:
                st.subheader("💡 Optimal Plan")
                m1, m2, m3 = st.columns(3)
                m1.metric("Basket Cost", f"{total} €")
                m2.metric("Total Savings", f"{savings} €", f"{savings_pct}%")
                m3.metric("Route", f"{combo[0]} + {combo[1]}")
                
                st.dataframe(plan_df.style.highlight_max(subset=['Savings'], color='#d4edda'), use_container_width=True)
                
                # Strategic Negotiator Tip
                top_deal = plan_df.iloc[plan_df['Savings'].idxmax()]
                st.info(f"**Negotiator Tip:** Your best deal is **{top_deal['Found']}** at **{top_deal['Store']}**. It drives the highest ROI for this trip.")

    # Export Section
    st.divider()
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Raw_Promos')
    st.download_button("📥 Download Full Promo Database", buffer.getvalue(), "promo_export.xlsx", "application/vnd.ms-excel")
