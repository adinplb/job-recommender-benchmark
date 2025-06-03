import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # Kept if needed for other parts or future
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer # For SBERT
import random
from nltk.tokenize import word_tokenize
import nltk # For NLTK download check

# --- Initial NLTK Download Check (do this once) ---
try:
    word_tokenize("test")
except LookupError:
    with st.spinner("NLTK 'punkt' tokenizer not found. Downloading..."):
        nltk.download('punkt', quiet=True)
    st.toast("'punkt' tokenizer downloaded for session.", icon="✅")
except Exception as e:
    st.error(f"Error during NLTK setup: {e}. Word tokenization might fail.")


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

@st.cache_data
def load_onet_data_from_url(url):
    try:
        df = pd.read_csv(url)
        expected_onet_cols = ['O*NET-SOC Code', 'Title', 'Description']
        missing_cols = [col for col in expected_onet_cols if col not in df.columns]
        if missing_cols:
            st.error(f"O*NET data from URL is missing expected columns: {', '.join(missing_cols)}. Found: {df.columns.tolist()}")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading O*NET data from {url}: {e}")
        return None

def preprocess_text_for_sbert(text):
    if not isinstance(text, str):
        return ""
    return text.lower().strip()

def denoise_text(text_content: str, method: str, 
                 del_ratio: float = 0.6, 
                 word_freq_dict: dict = None, 
                 freq_threshold: int = 100) -> str:
    if not isinstance(text_content, str) or not text_content.strip(): return "" 
    try: words = word_tokenize(text_content)
    except Exception: return text_content 
    if not words: return ""
    output_words = []
    if method == 'a':
        if not (0 <= del_ratio <= 1): del_ratio = 0.0 
        words_to_keep = list(words); num_to_delete = int(len(words_to_keep) * del_ratio)
        for _ in range(num_to_delete):
            if not words_to_keep: break
            words_to_keep.pop(random.randrange(len(words_to_keep)))
        output_words = words_to_keep
    elif method == 'b':
        if word_freq_dict is None: output_words = list(words) 
        else: output_words = [word for word in words if word_freq_dict.get(word.lower(), 0) <= freq_threshold]
    elif method == 'c':
        temp_words_for_c = []
        if word_freq_dict is None: temp_words_for_c = list(words)
        else: temp_words_for_c = [word for word in words if word_freq_dict.get(word.lower(), 0) <= freq_threshold]
        random.shuffle(temp_words_for_c)
        output_words = temp_words_for_c
    else: raise ValueError(f"Unknown denoising method: {method}")
    return " ".join(output_words)

# --- Classification Functions ---
def add_category_column_first_word(df_input):
    df = df_input.copy()
    # ... (implementation from previous response, ensure sbert_job_embedding column is added as None) ...
    if 'Title' not in df.columns:
        df['category'] = "Error: No Title Column"; df['onet_soc_code'] = "N/A"; df['onet_match_score'] = np.nan
    else:
        df['category'] = df['Title'].apply(lambda x: x.split()[0] if isinstance(x, str) and x.strip() else 'Unknown (First Word)')
        df['onet_soc_code'] = 'N/A'; df['onet_match_score'] = np.nan
    df['sbert_job_embedding'] = [None] * len(df) # Ensure column exists for schema consistency
    df['final_noisy_text_jobs'] = [None] * len(df)
    return df


@st.cache_resource # Cache SBERT model loading
def get_sbert_model(model_name='all-MiniLM-L6-v2'):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading SBERT model '{model_name}': {e}. Ensure 'sentence-transformers' is installed.")
        return None

@st.cache_data(show_spinner=False)
def classify_with_sbert_and_tsdae_noise(
    _df_jobs, _onet_df, 
    deletion_ratio_a, freq_threshold_bc, 
    job_title_col='Title' # For error checking if Title is missing
):
    df_jobs_classified = _df_jobs.copy()
    onet_title_col = 'Title'
    onet_code_col = 'O*NET-SOC Code'
    onet_desc_col = 'Description'

    if not all(col in _onet_df.columns for col in [onet_title_col, onet_code_col, onet_desc_col]):
        st.error(f"O*NET data is missing required columns. Expected: '{onet_title_col}', '{onet_code_col}', '{onet_desc_col}'.")
        df_jobs_classified['category'] = 'Error: O*NET cols missing'; df_jobs_classified['onet_soc_code'] = 'Error'
        df_jobs_classified['onet_match_score'] = np.nan; df_jobs_classified['sbert_job_embedding'] = [None] * len(df_jobs_classified)
        df_jobs_classified['final_noisy_text_jobs'] = [None] * len(df_jobs_classified)
        return df_jobs_classified
    if job_title_col not in df_jobs_classified.columns: # Essential for basic processing
        st.error(f"Job data is missing '{job_title_col}' column.")
        df_jobs_classified['category'] = 'Error: Job Title col missing'; df_jobs_classified['onet_soc_code'] = 'Error'
        # ... (fill other columns for schema consistency)
        return df_jobs_classified

    # --- O*NET Data Preparation ---
    _onet_df['combined_onet_text'] = _onet_df[onet_title_col].fillna('').astype(str) + ' ' + _onet_df[onet_desc_col].fillna('').astype(str)
    _onet_df['processed_onet_text'] = _onet_df['combined_onet_text'].apply(preprocess_text_for_sbert)
    onet_texts_to_embed = _onet_df['processed_onet_text'].tolist()

    # --- Job Postings Data Preparation & TSDAE Noise ---
    st.session_state.current_status = "Preprocessing job posting data for TSDAE noise..."
    object_cols = df_jobs_classified.select_dtypes(include=['object']).columns
    df_jobs_classified['combined_job_text'] = df_jobs_classified[object_cols].fillna('').astype(str).agg(' '.join, axis=1)
    df_jobs_classified['processed_text_jobs'] = df_jobs_classified['combined_job_text'].apply(preprocess_text_for_sbert)

    all_words_for_freq = []
    for text_entry in df_jobs_classified['processed_text_jobs'].fillna('').astype(str):
        try: all_words_for_freq.extend([w.lower() for w in word_tokenize(text_entry)])
        except Exception: pass 
    word_freq_dict = pd.Series(all_words_for_freq).value_counts().to_dict() if all_words_for_freq else {}

    st.session_state.current_status = "Applying TSDAE Noise A (Deletion)..."
    df_jobs_classified['noisy_text_a'] = df_jobs_classified['processed_text_jobs'].apply(lambda x: denoise_text(x, method='a', del_ratio=deletion_ratio_a))
    st.session_state.current_status = "Applying TSDAE Noise B (High-Freq Removal)..."
    df_jobs_classified['noisy_text_b'] = df_jobs_classified['noisy_text_a'].apply(lambda x: denoise_text(x, method='b', word_freq_dict=word_freq_dict, freq_threshold=freq_threshold_bc))
    st.session_state.current_status = "Applying TSDAE Noise C (High-Freq Removal + Shuffle)..."
    df_jobs_classified['final_noisy_text_jobs'] = df_jobs_classified['noisy_text_b'].apply(lambda x: denoise_text(x, method='c', word_freq_dict=word_freq_dict, freq_threshold=freq_threshold_bc))
    
    job_texts_to_embed = df_jobs_classified['final_noisy_text_jobs'].tolist()

    sbert_model = get_sbert_model()
    if sbert_model is None: # Model loading failed
        df_jobs_classified['category'] = 'Error: SBERT model load'; df_jobs_classified['onet_soc_code'] = 'Error'
        return df_jobs_classified

    st.session_state.current_status = "Generating SBERT embeddings for NOISY job postings (can take minutes)..."
    job_embeddings = sbert_model.encode(job_texts_to_embed, show_progress_bar=False) # Progress managed by st.spinner
    st.session_state.current_status = "Generating SBERT embeddings for ORIGINAL O*NET texts..."
    onet_embeddings = sbert_model.encode(onet_texts_to_embed, show_progress_bar=False)
    
    df_jobs_classified['sbert_job_embedding'] = [emb.tolist() for emb in job_embeddings]

    st.session_state.current_status = "Calculating similarities and matching..."
    similarity_matrix = cosine_similarity(job_embeddings, onet_embeddings)
    matched_onet_titles, matched_onet_codes, match_scores = [], [], []
    for i in range(similarity_matrix.shape[0]):
        best_match_idx = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i, best_match_idx]
        matched_onet_titles.append(_onet_df.iloc[best_match_idx][onet_title_col])
        matched_onet_codes.append(_onet_df.iloc[best_match_idx][onet_code_col])
        match_scores.append(best_score)

    df_jobs_classified['category'] = matched_onet_titles
    df_jobs_classified['onet_soc_code'] = matched_onet_codes
    df_jobs_classified['onet_match_score'] = match_scores
    
    # Selectively drop intermediate columns, keep final_noisy_text for inspection
    cols_to_drop = ['combined_job_text', 'processed_text_jobs', 'noisy_text_a', 'noisy_text_b']
    df_jobs_classified = df_jobs_classified.drop(columns=[col for col in cols_to_drop if col in df_jobs_classified.columns], errors='ignore')
    return df_jobs_classified

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="TSDAE-SBERT Job Classifier")
st.title("🔧 TSDAE-SBERT Job Posting Classifier & Analyzer")
st.markdown("Classify ~4000 job postings using O*NET occupations via **TSDAE noised text + SBERT embeddings**.")
st.info(
    "**Note:** SBERT with TSDAE noise classification can take several minutes on the first run. "
    "Ensure `sentence-transformers` and `nltk` are installed (`pip install sentence-transformers nltk`)."
)

# --- Sidebar ---
st.sidebar.header("🛠️ Setup & Classification")

# Initialize session state for status updates
if 'current_status' not in st.session_state:
    st.session_state.current_status = "Ready"

status_placeholder = st.sidebar.empty() # For showing progress from cached function

with st.spinner("Loading main job postings data..."):
    df_loaded = load_data_from_url(DATA_URL, "Main Job Postings")

if df_loaded is None:
    st.error("Fatal Error: Main job data load failed. Dashboard cannot proceed.")
    st.stop()
st.sidebar.success(f"{df_loaded.shape[0]} job postings loaded.")

categorization_method = st.sidebar.radio(
    "Choose categorization method:",
    ("O*NET Classification (TSDAE & SBERT)", "First word of Title (Simple)"),
    key="categorization_method_radio"
)

onet_df = None
df_processed = None 
tsdae_deletion_ratio = 0.6
tsdae_freq_threshold = 100

if categorization_method == "O*NET Classification (TSDAE & SBERT)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("TSDAE Noise Parameters")
    tsdae_deletion_ratio = st.sidebar.slider("Deletion Ratio (Noise A)", 0.0, 1.0, 0.6, 0.05, key="tsdae_del_ratio")
    tsdae_freq_threshold = st.sidebar.slider("High Freq Threshold (Noise B & C)", 10, 500, 100, 10, key="tsdae_freq_thresh")
    
    with st.spinner("Loading O*NET standard occupation data..."):
        onet_df = load_onet_data_from_url(ONET_DATA_URL)
    
    if onet_df is not None:
        st.sidebar.success(f"O*NET data loaded: {onet_df.shape[0]} standard occupations.")
        if 'Title' in df_loaded.columns:
            spinner_text = "Applying TSDAE noise and SBERT classification... This may take several minutes."
            with st.spinner(spinner_text):
                st.session_state.current_status = "Starting classification..." # Initial status
                # Use a placeholder to show messages from the cached function
                # This is a bit of a workaround as st.write inside cached func doesn't update live
                # A more advanced way would use callbacks or session state for progress.
                df_processed = classify_with_sbert_and_tsdae_noise(
                    df_loaded, onet_df, 
                    deletion_ratio_a=tsdae_deletion_ratio, 
                    freq_threshold_bc=tsdae_freq_threshold
                )
                st.session_state.current_status = "Classification complete!" # Final status
            status_placeholder.info(st.session_state.current_status) # Show final status
            st.success("TSDAE-SBERT O*NET classification complete!")
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("SBERT O*NET Classification Sample:")
            sample_cols = ['Title', 'category', 'onet_soc_code', 'onet_match_score', 'final_noisy_text_jobs']
            display_sample_cols = [col for col in sample_cols if col in df_processed.columns]
            if display_sample_cols: st.sidebar.dataframe(df_processed[display_sample_cols].head(3))
            
            classified_scores = df_processed[df_processed['category'] != 'Unclassified (O*NET)']['onet_match_score'] # Should always be classified now
            avg_score_metric = f"{df_processed['onet_match_score'].mean():.3f}" if 'onet_match_score' in df_processed and not df_processed['onet_match_score'].empty else "N/A"
            st.sidebar.metric("Avg SBERT Match Score", avg_score_metric)
        else:
            st.error("Main job data is missing the 'Title' column. Cannot perform classification.")
            df_processed = add_category_column_first_word(df_loaded)
    else:
        st.sidebar.error("Failed to load O*NET data. TSDAE-SBERT O*NET classification unavailable.")
        df_processed = add_category_column_first_word(df_loaded)
        st.info("Fell back to 'First word of Title' categorization due to O*NET data issues.")

elif categorization_method == "First word of Title (Simple)":
    df_processed = add_category_column_first_word(df_loaded)
    st.info("Using 'First word of Title' for categorization.")


# --- Main dashboard rendering starts here, using df_processed ---
if df_processed is not None and 'category' in df_processed.columns and not (isinstance(df_processed['category'], pd.Series) and df_processed['category'].str.contains("Error|Pending", na=False).all()):
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Display & Filter Options")
    # (Filter, Display, Download, Analysis, and Canvas sections as in the previous version)
    # Ensure they use df_processed and df_working_set correctly.
    # The canvas plot will use 'sbert_job_embedding' column if available.

    unique_categories_processed = ['All'] + sorted(df_processed['category'].dropna().unique().tolist())
    selected_category_filter = st.sidebar.selectbox("Filter by Main Category:", unique_categories_processed, key="category_filter_select")

    if selected_category_filter != 'All':
        df_working_set = df_processed[df_processed['category'] == selected_category_filter].copy()
    else:
        df_working_set = df_processed.copy()

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Keyword Search")
    search_term = st.sidebar.text_input("Search keyword:", key="search_term_input")
    searchable_cols_default = ['Title', 'Job.Description', 'Description', 'category', 'final_noisy_text_jobs'] # Add relevant text cols
    available_for_search = [col for col in df_working_set.columns if df_working_set[col].dtype == 'object' and col not in ['sbert_job_embedding', 'combined_job_text', 'processed_text_jobs', 'noisy_text_a', 'noisy_text_b']]
    default_search_cols = [col for col in searchable_cols_default if col in available_for_search]
    if not default_search_cols and available_for_search: default_search_cols = available_for_search[:1]
    columns_to_search = st.sidebar.multiselect("Search in columns:", options=available_for_search, default=default_search_cols, key="search_cols_multiselect")

    if search_term and columns_to_search:
        search_mask = df_working_set[columns_to_search].astype(str).apply(lambda col: col.str.contains(search_term, case=False, na=False)).any(axis=1)
        df_working_set = df_working_set[search_mask]
    
    st.sidebar.markdown("---")
    st.sidebar.header("📄 Table Display")
    show_full_data_rows = st.sidebar.checkbox("Display all filtered rows", value=False, key="show_all_rows_checkbox")
    num_rows_preview = 20
    if not show_full_data_rows: num_rows_preview = st.sidebar.slider("Rows in preview table", 5, 200, 20, key="rows_slider_v3")
    
    all_available_cols_for_table = [col for col in df_working_set.columns if col not in ['sbert_job_embedding', 'combined_job_text', 'processed_text_jobs', 'noisy_text_a', 'noisy_text_b']]
    default_cols_for_table = ['Title', 'Position', 'Company', 'category', 'onet_soc_code', 'onet_match_score', 'final_noisy_text_jobs']
    actual_default_cols_for_table = [col for col in default_cols_for_table if col in all_available_cols_for_table]
    selected_cols_for_main_table = st.sidebar.multiselect("Columns for main table:", options=all_available_cols_for_table, default=actual_default_cols_for_table, key="table_cols_multiselect_v3")
    if not selected_cols_for_main_table and all_available_cols_for_table:
         selected_cols_for_main_table = [col for col in ['Title', 'category'] if col in all_available_cols_for_table] or all_available_cols_for_table[:1]

    st.sidebar.markdown("---")
    st.sidebar.header("📥 Download Filtered Data")
    df_to_download_cols = [col for col in selected_cols_for_main_table if col in df_working_set.columns and col != 'sbert_job_embedding'] # Exclude embedding col
    df_to_download = df_working_set[df_to_download_cols] if df_to_download_cols else df_working_set.drop(columns=['sbert_job_embedding'], errors='ignore')
    csv_to_download = convert_df_to_csv_cached(df_to_download) # Uses cached function from previous script
    st.sidebar.download_button(label="Download current data as CSV", data=csv_to_download, file_name='tsdae_sbert_classified_jobs.csv', mime='text/csv', key="download_csv_button_v2")

    # --- Main Panel Display ---
    st.header(f"📋 Displaying Job Postings ({df_working_set.shape[0]} entries match filters)")
    if selected_cols_for_main_table and not df_working_set.empty:
        st.dataframe(df_working_set[selected_cols_for_main_table].head(num_rows_preview if not show_full_data_rows else len(df_working_set)), height=(600 if show_full_data_rows else None))
    elif df_working_set.empty: st.info("No job postings match the current filter criteria.")
    else: st.warning("No columns selected or available for the main table display.")

    st.header("📈 Analysis of Displayed Data")
    if df_working_set.empty or 'category' not in df_working_set.columns:
        st.warning("No data or 'category' column available for analysis.")
    else:
        # Analysis sections (Category Distribution, Secondary Feature)
        # ... (Copy from previous version, ensuring they use df_working_set) ...
        col_analysis1, col_analysis2 = st.columns(2)
        with col_analysis1:
            st.subheader("Main Category Distribution")
            main_category_counts_filtered = df_working_set['category'].value_counts()
            if not main_category_counts_filtered.empty:
                num_top_main_cat_chart = st.slider("Top N main categories (chart):", 1, max(1, min(30, len(main_category_counts_filtered))), min(10, max(1, len(main_category_counts_filtered))), key="main_cat_chart_slider_v4")
                st.bar_chart(main_category_counts_filtered.nlargest(num_top_main_cat_chart))
        with col_analysis2:
            st.subheader("Value Counts (Main Categories)")
            if not main_category_counts_filtered.empty:
                num_top_main_cat_table = st.slider("Top N main categories (table):", 1, max(1, min(30, len(main_category_counts_filtered))), min(10, max(1, len(main_category_counts_filtered))), key="main_cat_table_slider_v4")
                st.dataframe(main_category_counts_filtered.nlargest(num_top_main_cat_table).reset_index().rename(columns={'index':'Category', 'category':'Count'}))
        
        st.markdown("---")
        st.subheader("Distribution of Another Feature")
        # ... (Secondary feature analysis logic as before)
        potential_secondary_cols = [col for col in df_working_set.columns if df_working_set[col].dtype == 'object' and df_working_set[col].nunique() < 70 and df_working_set[col].nunique() > 1 and col not in ['Title', 'Job.Description', 'Description', 'category', 'Company', 'onet_soc_code', 'sbert_job_embedding', 'combined_job_text', 'processed_text_jobs', 'noisy_text_a', 'noisy_text_b', 'final_noisy_text_jobs']]
        for known_cat_col in ['Position', 'Employment.Type', 'Industry']:
            if known_cat_col in df_working_set.columns and known_cat_col not in potential_secondary_cols:
                if df_working_set[known_cat_col].nunique() < 70 and df_working_set[known_cat_col].nunique() > 1: potential_secondary_cols.insert(0, known_cat_col)
        potential_secondary_cols = sorted(list(set(potential_secondary_cols)))

        if potential_secondary_cols:
            selected_secondary_col = st.selectbox("Select feature for distribution analysis:", options=potential_secondary_cols, index=0 if potential_secondary_cols else -1, key="secondary_col_selectbox_v3")
            if selected_secondary_col and selected_secondary_col in df_working_set.columns:
                secondary_col_counts = df_working_set[selected_secondary_col].value_counts()
                st.bar_chart(secondary_col_counts)
                if st.checkbox(f"Show value counts table for '{selected_secondary_col}'", key=f"table_for_{selected_secondary_col}_v4"):
                    st.dataframe(secondary_col_counts.reset_index().rename(columns={'index':selected_secondary_col, selected_secondary_col:'Count'}))
        else: st.write("No suitable columns for secondary distribution analysis in current filtered data.")


        # --- Canvas Visualization Section (Uses SBERT embeddings if available) ---
        st.markdown("---")
        st.header("🎨 Visual Canvas of Job Categories (PCA on SBERT Embeddings)")
        if df_working_set.empty or 'sbert_job_embedding' not in df_working_set.columns or df_working_set['sbert_job_embedding'].apply(lambda x: x is None).all() or len(df_working_set.dropna(subset=['sbert_job_embedding'])) < 2:
            st.warning("Not enough data or no SBERT embeddings in current selection for canvas visualization (need at least 2 postings with embeddings).")
        elif 'category' not in df_working_set.columns or df_working_set['category'].dropna().empty:
            st.warning("No 'category' data available for coloring the canvas visualization in the current selection.")
        else:
            try:
                # Use a placeholder for status updates during plot generation
                canvas_status = st.empty()
                canvas_status.info("Preparing data for SBERT canvas visualization...")

                plot_data_sbert = df_working_set.dropna(subset=['sbert_job_embedding', 'category']).copy()
                
                if len(plot_data_sbert) < 2:
                     st.warning("Not enough valid SBERT embeddings (after dropping NaNs) in current selection for PCA plot (need at least 2).")
                else:
                    # sbert_job_embedding column now contains lists of floats
                    sbert_embeddings_for_plot = np.array(plot_data_sbert['sbert_job_embedding'].tolist())
                    
                    canvas_status.info("Performing PCA on SBERT embeddings...")
                    pca_sbert_plot = PCA(n_components=2, random_state=42)
                    reduced_sbert_features = pca_sbert_plot.fit_transform(sbert_embeddings_for_plot)
                    
                    plot_df_sbert = pd.DataFrame({
                        'pca_x': reduced_sbert_features[:,0], 
                        'pca_y': reduced_sbert_features[:,1], 
                        'category': plot_data_sbert['category'], 
                        'title': plot_data_sbert['Title'] 
                    })
                    
                    canvas_status.info("Generating plot...")
                    fig_canvas_sbert, ax_canvas_sbert = plt.subplots(figsize=(12, 9))
                    unique_cats_sbert_series = plot_df_sbert['category'].astype('category')
                    cat_codes_sbert = unique_cats_sbert_series.cat.codes
                    num_unique_cats_sbert = len(unique_cats_sbert_series.cat.categories)
                    
                    cmap_sbert_name = 'tab10' if num_unique_cats_sbert <= 10 else ('tab20' if num_unique_cats_sbert <= 20 else 'viridis')
                    cmap_sbert = plt.get_cmap(cmap_sbert_name, num_unique_cats_sbert if num_unique_cats_sbert > 0 else 1)

                    ax_canvas_sbert.scatter(plot_df_sbert['pca_x'], plot_df_sbert['pca_y'], c=cat_codes_sbert, cmap=cmap_sbert, alpha=0.6, s=50, edgecolor='k', linewidths=0.3)
                    ax_canvas_sbert.set_title('Job Postings by O*NET Category (SBERT Embeddings + PCA)', fontsize=15)
                    ax_canvas_sbert.set_xlabel('PCA Component 1 (from SBERT)', fontsize=12); ax_canvas_sbert.set_ylabel('PCA Component 2 (from SBERT)', fontsize=12)
                    ax_canvas_sbert.grid(True, linestyle='--', alpha=0.5)

                    if num_unique_cats_sbert > 0:
                        legend_handles_sbert = [plt.Line2D([0], [0], marker='o', color='w', label=str(cat_name), markerfacecolor=cmap_sbert(i / (num_unique_cats_sbert - 1 if num_unique_cats_sbert > 1 else 1.0)), markersize=8) for i, cat_name in enumerate(unique_cats_sbert_series.cat.categories)]
                        if legend_handles_sbert: ax_canvas_sbert.legend(handles=legend_handles_sbert, title='O*NET Categories', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                    st.pyplot(fig_canvas_sbert)
                    canvas_status.empty() # Clear status message
                    if st.checkbox("Show sample titles for SBERT canvas points (first 10)?", key="show_sbert_canvas_sample_v2"): st.dataframe(plot_df_sbert[['title', 'category']].head(10))
            except Exception as e_sbert_plot: 
                st.error(f"Could not generate SBERT canvas: {e_sbert_plot}")
                canvas_status.empty()
else:
    if df_loaded is not None and ('Title' not in df_loaded.columns):
        st.error("The loaded job data does not contain a 'Title' column, which is essential for all categorization methods.")
    else:
        st.warning("Data processing is incomplete or categorization failed. Please select a valid categorization method and ensure all required data (like O*NET from URL) is loaded and processed successfully.")

st.sidebar.markdown("---")
st.sidebar.info("Adjust options in the sidebar to filter and explore job postings.")
