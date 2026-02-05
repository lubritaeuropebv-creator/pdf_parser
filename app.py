import streamlit as st
import pandas as pd
import json, gc
import google.generativeai as genai
from rapidfuzz import process, fuzz
import itertools

# --- 1. AI RECIPE GENERATOR ---
def ai_generate_recipe(api_key, idea):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")
    
    prompt = f"""
    User Idea: {idea}
    Task: Create a recipe and a shopping list.
    Return ONLY a JSON object:
    {{
      "recipe_name": "Name",
      "ingredients": ["pienas", "kiaušiniai", "miltai"], 
      "instructions": "Step 1, Step 2..."
    }}
    Note: Ingredients must be in Lithuanian to match the flyers.
    """
    try:
        response = model.generate_content(prompt)
        clean_json = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json)
    except: return None

# --- 2. MULTI-SHOP BASKET OPTIMIZER ---
def optimize_basket(ingredients, df, max_shops):
    stores = df['store'].unique()
    best_total = float('inf')
    best_plan = None
    missing_items = []

    # Find the best combination of N stores
    for combo in itertools.combinations(stores, max_shops):
        temp_subset = df[df['store'].isin(combo)]
        current_total = 0
        current_items = []
        found_ingredients = set()

        for ing in ingredients:
            # Fuzzy match recipe ingredient to flyer product names
            choices = temp_subset['product_name'].tolist()
            match = process.extractOne(ing, choices, scorer=fuzz.WRatio)
            
            if match and match[1] > 60:
                row = temp_subset[temp_subset['product_name'] == match[0]].iloc[0]
                current_total += row['disc_price']
                current_items.append({
                    "Ingredient": ing,
                    "Found As": row['product_name'],
                    "Store": row['store'],
                    "Price": f"{row['disc_price']:.2f}€"
                })
                found_ingredients.add(ing)
        
        # We want the combination that finds the MOST ingredients at the LOWEST price
        if len(found_ingredients) >= len(ingredients) * 0.5: # At least 50% found
            if current_total < best_total:
                best_total = current_total
                best_plan = (combo, current_items, current_total)
                missing_items = [i for i in ingredients if i not in found_ingredients]

    return best_plan, missing_items

# --- 3. UI IMPLEMENTATION ---
st.title("👨‍🍳 AI Chef & Multi-Shop Procurement")

if 'master_df' in st.session_state and not st.session_state['master_df'].empty:
    df = st.session_state['master_df']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. What's the plan?")
        user_idea = st.text_input("Enter a dish or idea:", "Soti vakarienė šeimai")
        shop_count = st.slider("Max stores to visit:", 1, 5, 2)
        exec_btn = st.button("🪄 Calculate Best Route & Recipe")

    if exec_btn:
        recipe = ai_generate_recipe(st.session_state.get('api_key', ""), user_idea)
        
        if recipe:
            with col2:
                st.subheader(f"🍴 {recipe['recipe_name']}")
                st.write(recipe['instructions'])
            
            st.divider()
            
            # Run Optimizer
            plan, missing = optimize_basket(recipe['ingredients'], df, shop_count)
            
            if plan:
                combo, items, total = plan
                st.subheader(f"💰 Optimized Basket: {total:.2f}€")
                st.info(f"**Route:** {' ➡ '.join(combo)}")
                st.table(pd.DataFrame(items))
                
                if missing:
                    st.warning(f"⚠️ **Not in flyers (Full Price needed):** {', '.join(missing)}")
            else:
                st.error("Could not find enough ingredients in the current flyers. Try adding more shops!")

else:
    st.warning("Please upload flyer PDFs in the sidebar first to build your price database.")
