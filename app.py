# --- 1. DOWNLOAD UTILITY (at the top of your script) ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 2. THE SIDEBAR EXPORT HUB ---
with st.sidebar:
    st.header("🔑 Authentication")
    api_key = st.text_input("Gemini API Key", type="password")
    
    # NEW: Global Download Button that persists
    if not st.session_state['master_df'].empty:
        st.divider()
        st.subheader("📥 Global Export")
        current_csv = convert_df_to_csv(st.session_state['master_df'])
        
        # Generate a dynamic filename based on the shops currently in the DB
        shops_in_db = st.session_state['master_df']['store'].unique()
        shop_tags = "_".join(shops_in_db)
        timestamp = pd.Timestamp.now().strftime("%H%M")
        
        st.download_button(
            label=f"Download Data ({len(shops_in_db)} Shops)",
            data=current_csv,
            file_name=f"LT_Market_{shop_tags}_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary" # Highlighted so you don't forget to save
        )
        if st.button("🗑️ Clear Database"):
            st.session_state['master_df'] = pd.DataFrame()
            st.rerun()

    st.divider()
    st.header("📄 Flyer Upload")
    shop = st.selectbox("Select Retailer", ["Lidl", "Maxima", "IKI", "Rimi", "Norfa"])
    file = st.file_uploader(f"Upload {shop} PDF", type="pdf")
    
    if st.button("🚀 Process & Add"):
        if api_key and file:
            new_data = process_flyer(api_key, file, shop)
            if not new_data.empty:
                # Merge and keep unique products
                st.session_state['master_df'] = pd.concat(
                    [st.session_state['master_df'], new_data], 
                    ignore_index=True
                ).drop_duplicates()
                st.success(f"Added {shop}! The download button above is updated.")
                st.rerun() # Refresh to update the download button immediately
