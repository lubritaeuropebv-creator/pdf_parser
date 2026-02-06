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
    tab1, tab2, tab3 = st.tabs(["📊 Rinkos apžvalga", "🍳 AI Receptai ir krepšelis", "🛒 Greitas prekių krepšelis"])
    
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
            # --- ŽMONIŲ SKAIČIAUS PASIRINKIMAS ---
            asmenu_skaicius = st.slider("Kiek asmenų gaminsite?", 1, 10, 2)
            
            with st.spinner("🧠 AI strategas analizuoja rinkos duomenis ir grupines nuolaidas..."):
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-3-flash-preview")
                
                # Paverčiame prekių duomenų bazę į JSON formatą Gemini analizei
                rinkos_kontekstas = st.session_state['master_df'][['product_name', 'package_size', 'disc_price', 'std_price', 'discount_pct', 'store']].to_json(orient='records')

                # 1. AI STRATEGIJOS PROMPTAS (Su grupinėmis nuolaidomis)
                strategijos_promptas = f"""
                Tu esi profesionalus pirkimų strategas. 
                Užduotys:
                1. Sukurk receptą patiekalui: '{user_input}', apskaičiuotą {asmenu_skaicius} asmenims.
                2. Surask geriausiai tinkančius ingredientus pateiktuose RINKOS DUOMENYSE.
                3. SVARBU: Atpažink grupines nuolaidas (pvz., jei recepte yra agurkai, o duomenyse yra 'Ilgavaisiai agurkai' arba nuolaida visai kategorijai).
                4. Įtrauk 'package_size' (pakuotės dydį) ir tikslią nuolaidą % (jei yra).
                5. Parink pigiausią įmanomą atitikmenį visose parduotuvėse.

                RINKOS DUOMENYS:
                {rinkos_kontekstas}

                Atsakymą pateik JSON formatu (lietuvių kalba):
                {{
                  "recepto_pavadinimas": "string",
                  "instrukcijos": "string",
                  "pigiausias_krepšelis": [
                    {{
                      "ingredientas": "pavadinimas", 
                      "preke": "pavadinimas iš DB", 
                      "dydis": "pakuotės dydis", 
                      "kaina": 0.0, 
                      "standartine_kaina": 0.0, 
                      "parduotuve": "Parduotuvė", 
                      "nuolaida": "pvz. 40%"
                    }}
                  ],
                  "vieno_sustojimo_parinktys": [
                    {{"parduotuve": "string", "bendra_kaina": 0.0, "prekiu_atitikimas_proc": 0}}
                  ]
                }}
                """
                
                try:
                    resp = model.generate_content(strategijos_promptas)
                    clean_json = resp.text.strip().replace("```json", "").replace("```", "")
                    strat_duomenys = json.loads(clean_json)
                    
                    st.success(f"🍽️ **Strategija patiekalui:** {strat_duomenys['recepto_pavadinimas']} ({asmenu_skaicius} asm.)")
                    with st.expander("👨‍🍳 Gaminimo eiga"):
                        st.write(strat_duomenys['instrukcijos'])

                    # --- 2. STRATEGINĖS PARINKTYS ---
                    tab_hustle, tab_one_stop = st.tabs(["🏃 Pigiausias krepšelis", "🏠 Vieno sustojimo pirkimas"])

                    with tab_hustle:
                        h_df = pd.DataFrame(strat_duomenys['pigiausias_krepšelis'])
                        pigiausia_suma = h_df['kaina'].sum()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Minimali kaina", f"{pigiausia_suma:.2f}€")
                        with col2:
                            std_sum = h_df[h_df['standartine_kaina'] > 0]['standartine_kaina'].sum()
                            disc_sum = h_df[h_df['standartine_kaina'] > 0]['kaina'].sum()
                            sutaupymas = std_sum - disc_sum
                            st.metric("Sutaupyta", f"{max(0, sutaupymas):.2f}€")

                        st.markdown("#### 🛒 Pirkinių sąrašas (su pakuočių dydžiais)")
                        
                        # --- ATNAUJINTAS FORMATAVIMAS (2 skaitmenys po kablelio) ---
                        display_df = h_df[['ingredientas', 'preke', 'dydis', 'standartine_kaina', 'kaina', 'nuolaida', 'parduotuve']].copy()
                        
                        # Suformatuojame kainų stulpelius kaip tekstą su dviem nuliais
                        display_df['standartine_kaina'] = display_df['standartine_kaina'].apply(lambda x: f"{x:.2f}€" if x > 0 else "Nėra")
                        display_df['kaina'] = display_df['kaina'].apply(lambda x: f"{x:.2f}€")
                        
                        st.table(display_df)

                        # --- SMS SĄRAŠAS ---
                        st.divider()
                        sms_tekstas = f"🛒 PIRKINIŲ SĄRAŠAS ({strat_duomenys['recepto_pavadinimas']}):\n"
                        for _, row in h_df.iterrows():
                            # SMS sąraše taip pat užtikriname 2 skaitmenis
                            sms_tekstas += f"• {row['preke']} ({row['dydis']}) - {row['kaina']:.2f}€ @ {row['parduotuve']}\n"
                        sms_tekstas += f"\nVISO: {pigiausia_suma:.2f}€"
                        
                        st.subheader("📱 Kopijuoti į telefoną")
                        st.text_area("SMS / Messenger paruoštukas:", value=sms_tekstas, height=180)

                    with tab_one_stop:
                        s_df = pd.DataFrame(strat_duomenys['vieno_sustojimo_parinktys']).sort_values(by="bendra_kaina")
                        st.table(s_df)
                        
                        priemoka = s_df.iloc[0]['bendra_kaina'] - pigiausia_suma
                        st.info(f"💡 **Patogumo kaina:** Perkant tik {s_df.iloc[0]['parduotuve']}, permokėsite **{priemoka:.2f}€**.")

                except Exception as e:
                    st.error(f"AI strategijos klaida: {str(e)}")
    with tab3:
        st.header("🛒 Greitas produktų krepšelio parinkimas")
        st.info("Įveskite produktus kableliais (pvz.: pienas, agurkai, sausainiai).")
        
        prekiu_ivestis = st.text_area("Jūsų pirkinių sąrašas:", placeholder="Pvz.: Pienas, kiaušiniai, agurkai")
        vykdyti_paieska = st.button("Surasti geriausias kainas")
    
        if vykdyti_paieska and prekiu_ivestis:
            if st.session_state['master_df'].empty:
                st.warning("Klaida: Duomenų bazė tuščia. Įkelkite skrajutes šoninėje juostoje!")
            else:
                with st.spinner("AI analizuoja kainas ir grupines nuolaidas..."):
                    ieskomos_prekes = [p.strip() for p in prekiu_ivestis.split(',') if p.strip()]
                    df = st.session_state['master_df']
                    
                    # Paruošiame duomenų kontekstą (tik pirmas 500 eilučių, kad neviršytų atminties)
                    rinkos_json = df[['product_name', 'package_size', 'disc_price', 'std_price', 'discount_pct', 'store']].head(1000).to_json(orient='records')
                    
                    rezultatai = []
                    nerasta = []
    
                    for preke in ieskomos_prekes:
                        # AI užklausa kiekvienai prekei
                        paieskos_promptas = f"""
                        Tu esi lietuviškų pirkinių ekspertas. 
                        Rask geriausią atitikmenį produktui: '{preke}'.
                        
                        INSTRUKCIJOS:
                        1. Jei nėra konkretaus prekės ženklo, rinkis bet kurį tos kategorijos produktą (pvz. 'pienas' -> bet koks pienas).
                        2. Prioritetą teik mažiausiai 'disc_price'.
                        3. BŪTINAI įtrauk 'package_size'.
                        
                        DUOMENYS: {rinkos_json}
                        
                        Grąžink TIK JSON:
                        {{
                            "preke": "pavadinimas iš DB",
                            "dydis": "pakuotė",
                            "kaina": 0.0,
                            "std_kaina": 0.0,
                            "nuolaida": "proc",
                            "parduotuve": "pavadinimas"
                        }}
                        """
                        
                        try:
                            r = model.generate_content(paieskos_promptas)
                            # Išvalome atsakymą nuo galimų Markdown šiukšlių
                            cleaned_res = r.text.strip().replace("```json", "").replace("```", "")
                            res = json.loads(cleaned_res)
                            
                            if res and "preke" in res and res["preke"]:
                                rezultatai.append(res)
                            else:
                                # AUTOMATINIS SAUGIKLIS: Jei AI nerado, ieškome tiesiogiai pagal tekstą
                                fallback = df[df['product_name'].str.contains(preke, case=False, na=False)].sort_values(by='disc_price')
                                if not fallback.empty:
                                    f_row = fallback.iloc[0]
                                    rezultatai.append({
                                        "preke": f_row['product_name'],
                                        "dydis": f_row['package_size'],
                                        "kaina": f_row['disc_price'],
                                        "std_kaina": f_row['std_price'],
                                        "nuolaida": f"{f_row['discount_pct']:.0f}%",
                                        "parduotuve": f_row['store']
                                    })
                                else:
                                    nerasta.append(preke)
                        except Exception:
                            nerasta.append(preke)
    
                    # --- REZULTATŲ ATVAIZDAVIMAS ---
                    if rezultatai:
                        res_df = pd.DataFrame(rezultatai)
                        viso_suma = res_df['kaina'].sum()
                        
                        c1, c2 = st.columns(2)
                        c1.metric("Krepšelio suma", f"{viso_suma:.2f}€")
                        
                        # Formatuojame kainas su 2 skaitmenimis
                        display_res = res_df.copy()
                        display_res['kaina'] = display_res['kaina'].apply(lambda x: f"{x:.2f}€")
                        display_res['std_kaina'] = display_res['std_kaina'].apply(lambda x: f"{x:.2f}€" if x > 0 else "Nėra")
                        
                        st.table(display_res[['preke', 'dydis', 'std_kaina', 'kaina', 'nuolaida', 'parduotuve']])
                        
                        # SMS Paruoštukas
                        sms = f"🛒 PIGIAUSIAS KREPŠELIS:\n"
                        for _, row in res_df.iterrows():
                            sms += f"• {row['preke']} ({row['dydis']}) - {row['kaina']:.2f}€ @ {row['parduotuve']}\n"
                        sms += f"\nVISO: {viso_suma:.2f}€"
                        st.text_area("SMS nukopijavimui:", value=sms, height=180)
                    
                    if nerasta:
                        st.warning(f"Nepavyko surasti pasiūlymų šioms prekėms: {', '.join(nerasta)}")
else:
    st.info("Upload retailer flyers to enable the Kitchen Strategist.")
