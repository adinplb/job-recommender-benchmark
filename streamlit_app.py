import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # Kept for TF-IDF canvas as an option, or remove if SBERT canvas is sole focus
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer # For SBERT

# URL for the main job postings dataset
DATA_URL = 'https://raw.githubusercontent.com/adinplb/largedataset-JRec/refs/heads/main/Filtered_Jobs_4000.csv'
# URL for the O*NET Occupation Data
ONET_DATA_URL = 'https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Occupation%20Data.csv'

# --- Helper Functions ---
@st.cache_data
def load_data_from_url(url, data_name="Data"):
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading {data_name} from {url}: {e}")
        return None

def preprocess_text_for_sbert(text):
    if not isinstance(text, str):
        return ""
    return text.lower().strip() # SBERT often benefits from minimal preprocessing

# --- Classification Functions ---
def add_category_column_first_word(df_input):
    df = df_input.copy()
    if 'Title' not in df.columns:
        st.error("Error: 'Title' column not found. Cannot create 'first word' categories.")
        df['category'] = "Error: No Title Column"
        df['onet_soc_code'] = "N/A"
        df['onet_match_score'] = np.nan
        df['sbert_job_embedding'] = None # For schema consistency
        return df
        
    df['category'] = df['Title'].apply(
        lambda x: x.split()[0] if isinstance(x, str) and x.strip() else 'Unknown (First Word)'
    )
    df['onet_soc_code'] = 'N/A'
    df['onet_match_score'] = np.nan
    df['sbert_job_embedding'] = [None] * len(df) # Placeholder
    return df

@st.cache_resource # Cache the SBERT model loading
def get_sbert_model(model_name='all-MiniLM-L6-v2'):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading SBERT model '{model_name}': {e}. Please ensure 'sentence-transformers' is installed.")
        return None

@st.cache_data(show_spinner=False) # Show custom spinner in the calling function
def classify_with_sbert(_df_jobs, _onet_df, job_text_col='processed_job_text', onet_text_col='processed_onet_text'):
    """
    Classifies job postings using SBERT embeddings and cosine similarity.
    Ensures all jobs are classified.
    Returns DataFrame with new columns: 'category', 'onet_soc_code', 'onet_match_score', 'sbert_job_embedding'.
    """
    df_jobs_classified = _df_jobs.copy()
    
    # Expected O*NET columns (user confirmed these are in their ONET_DATA_URL file)
    onet_title_col = 'Title'
    onet_code_col = 'O*NET-SOC Code'
    onet_desc_col = 'Description'

    if not all(col in _onet_df.columns for col in [onet_title_col, onet_code_col, onet_desc_col]):
        st.error(f"O*NET data is missing one or more required columns: '{onet_title_col}', '{onet_code_col}', '{onet_desc_col}'.")
        df_jobs_classified['category'] = 'Error: O*NET cols missing'
        df_jobs_classified['onet_soc_code'] = 'Error'
        df_jobs_classified['onet_match_score'] = np.nan
        df_jobs_classified['sbert_job_embedding'] = [None] * len(df_jobs_classified)
        return df_jobs_classified

    # 1. Prepare text for SBERT
    # For Job Postings: Combine all object type columns
    st.write("Preprocessing job posting data for SBERT...")
    object_cols = df_jobs_classified.select_dtypes(include=['object']).columns
    df_jobs_classified['combined_job_text'] = df_jobs_classified[object_cols].fillna('').astype(str).agg(' '.join, axis=1)
    df_jobs_classified[job_text_col] = df_jobs_classified['combined_job_text'].apply(preprocess_text_for_sbert)

    # For O*NET Data: Combine 'Title' and 'Description'
    st.write("Preprocessing O*NET data for SBERT...")
    _onet_df['combined_onet_text'] = _onet_df[onet_title_col].fillna('').astype(str) + ' ' + _onet_df[onet_desc_col].fillna('').astype(str)
    _onet_df[onet_text_col] = _onet_df['combined_onet_text'].apply(preprocess_text_for_sbert)

    job_texts_to_embed = df_jobs_classified[job_text_col].tolist()
    onet_texts_to_embed = _onet_df[onet_text_col].tolist()

    if not job_texts_to_embed or not onet_texts_to_embed:
        st.error("No text data available for embedding after preprocessing.")
        df_jobs_classified['category'] = 'Error: No text for SBERT'
        # ... (fill other columns)
        return df_jobs_classified

    # 2. Load SBERT model
    sbert_model = get_sbert_model()
    if sbert_model is None:
        df_jobs_classified['category'] = 'Error: SBERT model load failed'
        # ... (fill other columns)
        return df_jobs_classified

    # 3. Generate Embeddings
    st.write("Generating SBERT embeddings for job postings (can take a few minutes for 4000 rows)...")
    job_embeddings = sbert_model.encode(job_texts_to_embed, show_progress_bar=True)
    st.write("Generating SBERT embeddings for O*NET occupations...")
    onet_embeddings = sbert_model.encode(onet_texts_to_embed, show_progress_bar=True)

    df_jobs_classified['sbert_job_embedding'] = list(job_embeddings) # Store embeddings

    # 4. Calculate Cosine Similarity & Match
    st.write("Calculating similarities and matching...")
    similarity_matrix = cosine_similarity(job_embeddings, onet_embeddings)

    matched_onet_titles = []
    matched_onet_codes = []
    match_scores = []

    for i in range(similarity_matrix.shape[0]):
        best_match_idx = np.argmax(similarity_matrix[i]) # Always find the best match
        best_score = similarity_matrix[i, best_match_idx]
        
        matched_onet_titles.append(_onet_df.iloc[best_match_idx][onet_title_col])
        matched_onet_codes.append(_onet_df.iloc[best_match_idx][onet_code_col])
        match_scores.append(best_score)

    df_jobs_classified['category'] = matched_onet_titles
    df_jobs_classified['onet_soc_code'] = matched_onet_codes
    df_jobs_classified['onet_match_score'] = match_scores
    
    return df_jobs_classified

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Job Classifier (SBERT & O*NET)")
st.title("✨ Advanced Job Posting Classifier (SBERT & O*NET)")
st.markdown(
    "Classify ~4000 job postings using O*NET occupations via SBERT embeddings, "
    "or use a simple 'first word of title' rule."
)
st.info(
    "**Note:** SBERT classification can take a few minutes on the first run for 4000 items. "
    "Ensure you have `sentence-transformers` installed (`pip install sentence-transformers`)."
)

# --- Sidebar ---
st.sidebar.header("🛠️ Setup & Classification")

with st.spinner("Loading main job postings data..."):
    df_loaded = load_data_from_url(DATA_URL, "Main Job Postings")

if df_loaded is None:
    st.error("Fatal Error: Could not load main job postings data. Dashboard cannot proceed.")
    st.stop()

st.sidebar.success(f"{df_loaded.shape[0]} job postings loaded.")

categorization_method = st.sidebar.radio(
    "Choose categorization method:",
    ("O*NET Standard Classification (SBERT)", "First word of Title"),
    key="categorization_method_radio"
)

onet_df = None
df_processed = None 

if categorization_method == "O*NET Standard Classification (SBERT)":
    with st.spinner("Loading O*NET standard occupation data from URL..."):
        onet_df = load_data_from_url(ONET_DATA_URL, "O*NET Occupations") # Using the generic loader
    
    if onet_df is not None:
        # Check for essential O*NET columns early
        expected_onet_cols = ['O*NET-SOC Code', 'Title', 'Description']
        if not all(col in onet_df.columns for col in expected_onet_cols):
            st.sidebar.error(f"O*NET data is missing required columns: {expected_onet_cols}. Cannot use for SBERT classification.")
            onet_df = None # Invalidate onet_df
        else:
             st.sidebar.success(f"O*NET data loaded: {onet_df.shape[0]} standard occupations.")

    if onet_df is not None and 'Title' in df_loaded.columns:
        with st.spinner("Performing SBERT-based O*NET classification... This may take several minutes for 4000 postings."):
            df_processed = classify_with_sbert(df_loaded, onet_df)
        st.success("SBERT-based O*NET classification complete!")
        st.sidebar.markdown("---")
        st.sidebar.subheader("SBERT O*NET Classification Sample:")
        sample_cols = ['Title', 'category', 'onet_soc_code', 'onet_match_score']
        display_sample_cols = [col for col in sample_cols if col in df_processed.columns]
        if display_sample_cols:
             st.sidebar.dataframe(df_processed[display_sample_cols].head(3))
        avg_score_metric = f"{df_processed['onet_match_score'].mean():.2f}" if 'onet_match_score' in df_processed and not df_processed['onet_match_score'].empty else "N/A"
        st.sidebar.metric("Avg SBERT Match Score", avg_score_metric)
    elif 'Title' not in df_loaded.columns:
        st.error("Main job data is missing the 'Title' column. Cannot perform classification.")
        df_processed = add_category_column_first_word(df_loaded) # Fallback sets error category
    else: # O*NET df is None
        st.sidebar.error("Failed to load or validate O*NET data. SBERT O*NET classification unavailable.")
        df_processed = add_category_column_first_word(df_loaded)
        st.info("Fell back to 'First word of Title' categorization due to O*NET data issues.")


elif categorization_method == "First word of Title":
    df_processed = add_category_column_first_word(df_loaded)
    st.info("Using 'First word of Title' for categorization.")


if df_processed is not None and 'category' in df_processed.columns and not (isinstance(df_processed['category'], pd.Series) and df_processed['category'].str.contains("Error|Pending", na=False).all()):
    # --- Remainder of Sidebar (Filters, Display, Download) ---
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Display & Filter Options")
    unique_categories_processed = ['All'] + sorted(df_processed['category'].dropna().unique().tolist())
    selected_category_filter = st.sidebar.selectbox("Filter by Main Category:", unique_categories_processed, key="category_filter_select")

    if selected_category_filter != 'All':
        df_working_set = df_processed[df_processed['category'] == selected_category_filter].copy()
    else:
        df_working_set = df_processed.copy()

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Keyword Search")
    search_term = st.sidebar.text_input("Search keyword in selected text columns:", key="search_term_input")
    available_for_search = [col for col in df_working_set.columns if df_working_set[col].dtype == 'object' and col not in ['sbert_job_embedding', 'combined_job_text', 'processed_job_text']] # Exclude helper columns
    default_search_cols = [col for col in ['Title', 'Job.Description', 'Description', 'category'] if col in available_for_search]
    if not default_search_cols and available_for_search: default_search_cols = available_for_search[:1]
    columns_to_search = st.sidebar.multiselect("Select columns for keyword search:", options=available_for_search, default=default_search_cols, key="search_cols_multiselect")

    if search_term and columns_to_search:
        search_mask = df_working_set[columns_to_search].astype(str).apply(lambda col: col.str.contains(search_term, case=False, na=False)).any(axis=1)
        df_working_set = df_working_set[search_mask]
    
    st.sidebar.markdown("---")
    st.sidebar.header("📄 Table Display")
    show_full_data_rows = st.sidebar.checkbox("Display all filtered rows in table", value=False, key="show_all_rows_checkbox")
    num_rows_preview = 20
    if not show_full_data_rows: num_rows_preview = st.sidebar.slider("Number of rows for preview table", 5, 200, 20, key="rows_slider_v2")
    
    all_available_cols_for_table = [col for col in df_working_set.columns if col not in ['sbert_job_embedding', 'combined_job_text', 'processed_job_text']]
    default_cols_for_table = ['Title', 'Position', 'Company', 'category', 'onet_soc_code', 'onet_match_score']
    actual_default_cols_for_table = [col for col in default_cols_for_table if col in all_available_cols_for_table]
    selected_cols_for_main_table = st.sidebar.multiselect("Select columns for main table display:", options=all_available_cols_for_table, default=actual_default_cols_for_table, key="table_cols_multiselect_v2")
    if not selected_cols_for_main_table and all_available_cols_for_table:
         selected_cols_for_main_table = [col for col in ['Title', 'category'] if col in all_available_cols_for_table] or all_available_cols_for_table[:1]

    st.sidebar.markdown("---")
    st.sidebar.header("📥 Download Filtered Data")
    @st.cache_data
    def convert_df_to_csv_cached(df_to_convert): return df_to_convert.to_csv(index=False).encode('utf-8')
    
    df_to_download_cols = [col for col in selected_cols_for_main_table if col in df_working_set.columns] # Ensure selected cols exist
    df_to_download = df_working_set[df_to_download_cols] if df_to_download_cols else df_working_set
    csv_to_download = convert_df_to_csv_cached(df_to_download)
    st.sidebar.download_button(label="Download current data as CSV", data=csv_to_download, file_name='sbert_classified_jobs.csv', mime='text/csv', key="download_csv_button")

    # --- Main Panel Display ---
    st.header(f"📋 Displaying Job Postings ({df_working_set.shape[0]} entries match filters)")
    if selected_cols_for_main_table and not df_working_set.empty:
        st.dataframe(df_working_set[selected_cols_for_main_table].head(num_rows_preview if not show_full_data_rows else len(df_working_set)), height=(600 if show_full_data_rows else None))
    elif df_working_set.empty:
        st.info("No job postings match the current filter criteria.")
    else: st.warning("No columns selected for the main table display or no columns available.")

    st.header("📈 Analysis of Displayed Data")
    if df_working_set.empty or 'category' not in df_working_set.columns:
        st.warning("No data or 'category' column available for analysis based on current filters.")
    else:
        # (Analysis sections for main category and secondary feature - largely unchanged)
        col_analysis1, col_analysis2 = st.columns(2)
        with col_analysis1:
            st.subheader("Main Category Distribution")
            main_category_counts_filtered = df_working_set['category'].value_counts()
            if not main_category_counts_filtered.empty:
                num_top_main_cat_chart = st.slider("Top N main categories (chart):", 1, max(1, min(30, len(main_category_counts_filtered))), min(10, max(1, len(main_category_counts_filtered))), key="main_cat_chart_slider_v3")
                st.bar_chart(main_category_counts_filtered.nlargest(num_top_main_cat_chart))
        with col_analysis2:
            st.subheader("Value Counts (Main Categories)")
            if not main_category_counts_filtered.empty:
                num_top_main_cat_table = st.slider("Top N main categories (table):", 1, max(1, min(30, len(main_category_counts_filtered))), min(10, max(1, len(main_category_counts_filtered))), key="main_cat_table_slider_v3")
                st.dataframe(main_category_counts_filtered.nlargest(num_top_main_cat_table).reset_index().rename(columns={'index':'Category', 'category':'Count'}))
        
        st.markdown("---")
        st.subheader("Distribution of Another Feature")
        potential_secondary_cols = [col for col in df_working_set.columns if df_working_set[col].dtype == 'object' and df_working_set[col].nunique() < 70 and df_working_set[col].nunique() > 1 and col not in ['Title', 'Job.Description', 'Description', 'category', 'Company', 'onet_soc_code', 'sbert_job_embedding', 'combined_job_text', 'processed_job_text']]
        for known_cat_col in ['Position', 'Employment.Type', 'Industry']: # Prioritize these
            if known_cat_col in df_working_set.columns and known_cat_col not in potential_secondary_cols:
                if df_working_set[known_cat_col].nunique() < 70 and df_working_set[known_cat_col].nunique() > 1: potential_secondary_cols.insert(0, known_cat_col)
        potential_secondary_cols = sorted(list(set(potential_secondary_cols))) # Deduplicate and sort

        if potential_secondary_cols:
            selected_secondary_col = st.selectbox("Select feature for distribution analysis:", options=potential_secondary_cols, index=0 if potential_secondary_cols else -1, key="secondary_col_selectbox_v2")
            if selected_secondary_col and selected_secondary_col in df_working_set.columns:
                secondary_col_counts = df_working_set[selected_secondary_col].value_counts()
                st.bar_chart(secondary_col_counts)
                if st.checkbox(f"Show value counts table for '{selected_secondary_col}'", key=f"table_for_{selected_secondary_col}_v3"):
                    st.dataframe(secondary_col_counts.reset_index().rename(columns={'index':selected_secondary_col, selected_secondary_col:'Count'}))
        else: st.write("No suitable columns for secondary distribution analysis in current filtered data.")

        # --- Canvas Visualization Section (Now uses SBERT embeddings if available) ---
        st.markdown("---")
        st.header("🎨 Visual Canvas of Job Categories (PCA on SBERT Embeddings)")
        st.markdown("This plot visualizes job postings in a 2D space based on their SBERT embeddings, colored by their assigned O*NET category.")

        if df_working_set.empty or 'sbert_job_embedding' not in df_working_set.columns or df_working_set['sbert_job_embedding'].isnull().all() or len(df_working_set) < 2:
            st.warning("Not enough data or no SBERT embeddings in current selection for canvas visualization (need at least 2 postings with embeddings).")
        elif 'category' not in df_working_set.columns or df_working_set['category'].dropna().empty:
            st.warning("No 'category' data available for coloring the canvas visualization in the current selection.")
        else:
            try:
                with st.spinner("Generating SBERT canvas visualization..."):
                    # Filter out rows where sbert_job_embedding is None (if any were not processed)
                    plot_data_sbert = df_working_set.dropna(subset=['sbert_job_embedding', 'category']).copy()
                    
                    if len(plot_data_sbert) < 2:
                         st.warning("Not enough valid SBERT embeddings in current selection for PCA plot (need at least 2).")
                    else:
                        sbert_embeddings_for_plot = np.array(plot_data_sbert['sbert_job_embedding'].tolist())
                        
                        pca_sbert_plot = PCA(n_components=2, random_state=42)
                        reduced_sbert_features = pca_sbert_plot.fit_transform(sbert_embeddings_for_plot)
                        
                        plot_df_sbert = pd.DataFrame({
                            'pca_x': reduced_sbert_features[:,0], 
                            'pca_y': reduced_sbert_features[:,1], 
                            'category': plot_data_sbert['category'], 
                            'title': plot_data_sbert['Title'] # Assuming 'Title' is still present
                        })
                        
                        fig_canvas_sbert, ax_canvas_sbert = plt.subplots(figsize=(12, 9))
                        unique_cats_sbert_series = plot_df_sbert['category'].astype('category')
                        cat_codes_sbert = unique_cats_sbert_series.cat.codes
                        num_unique_cats_sbert = len(unique_cats_sbert_series.cat.categories)
                        
                        cmap_sbert_name = 'tab10' if num_unique_cats_sbert <= 10 else ('tab20' if num_unique_cats_sbert <= 20 else 'viridis')
                        cmap_sbert = plt.get_cmap(cmap_sbert_name, num_unique_cats_sbert if num_unique_cats_sbert > 0 else 1)

                        ax_canvas_sbert.scatter(plot_df_sbert['pca_x'], plot_df_sbert['pca_y'], c=cat_codes_sbert, cmap=cmap_sbert, alpha=0.7, s=50, edgecolor='k', linewidths=0.5)
                        ax_canvas_sbert.set_title('Job Postings by O*NET Category (SBERT + PCA)', fontsize=15)
                        ax_canvas_sbert.set_xlabel('PCA Component 1 (from SBERT)', fontsize=12)
                        ax_canvas_sbert.set_ylabel('PCA Component 2 (from SBERT)', fontsize=12)
                        ax_canvas_sbert.grid(True, linestyle='--', alpha=0.5)

                        if num_unique_cats_sbert > 0:
                            legend_handles_sbert = [plt.Line2D([0], [0], marker='o', color='w', label=str(cat_name), markerfacecolor=cmap_sbert(i / (num_unique_cats_sbert - 1 if num_unique_cats_sbert > 1 else 1.0)), markersize=8) for i, cat_name in enumerate(unique_cats_sbert_series.cat.categories)]
                            if legend_handles_sbert: ax_canvas_sbert.legend(handles=legend_handles_sbert, title='O*NET Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
                        
                        plt.tight_layout(rect=[0, 0, 0.85, 1])
                        st.pyplot(fig_canvas_sbert)
                        if st.checkbox("Show sample titles for SBERT canvas points (first 10)?", key="show_sbert_canvas_sample"): st.dataframe(plot_df_sbert[['title', 'category']].head(10))
            except Exception as e_sbert_plot: st.error(f"Could not generate SBERT canvas: {e_sbert_plot}")
else:
    if df_loaded is not None and ('Title' not in df_loaded.columns):
        st.error("The loaded job data does not contain a 'Title' column, essential for categorization.")
    else:
        st.warning("Data processing is incomplete or categorization failed. Please check settings and data sources in the sidebar.")

st.sidebar.markdown("---")
st.sidebar.info("Adjust options in the sidebar to filter and explore job postings.")
