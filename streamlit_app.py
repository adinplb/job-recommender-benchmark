import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# URL for the main job postings dataset
DATA_URL = 'https://raw.githubusercontent.com/adinplb/largedataset-JRec/refs/heads/main/Filtered_Jobs_4000.csv'
# URL for the O*NET Occupation Data (provided by you)
ONET_DATA_URL = 'https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Occupation%20Data.csv'


# --- Data Loading and Processing Functions ---
@st.cache_data # Cache the main data loading
def load_data_from_url(url, data_name="Job Postings"):
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading {data_name} data from {url}: {e}")
        return None

@st.cache_data # Cache the O*NET data loading
def load_onet_data_from_url(url):
    """
    Loads O*NET data from the specified CSV file URL.
    """
    try:
        df = pd.read_csv(url)
        # Explicitly check for expected columns based on user confirmation
        expected_onet_cols = ['O*NET-SOC Code', 'Title', 'Description']
        missing_cols = [col for col in expected_onet_cols if col not in df.columns]
        if missing_cols:
            st.error(f"O*NET data from URL is missing expected columns: {', '.join(missing_cols)}. Found columns: {df.columns.tolist()}")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading O*NET data from {url}: {e}")
        return None

def add_category_column_first_word(df_input):
    df = df_input.copy()
    if 'Title' not in df.columns:
        st.error("Error: 'Title' column not found in main job data. Cannot create 'first word' categories.")
        df['category'] = "Error: No Title Column"
        df['onet_soc_code'] = "N/A"
        df['onet_match_score'] = np.nan
        return df
        
    df['category'] = df['Title'].apply(
        lambda x: x.split()[0] if isinstance(x, str) and x.strip() else 'Unknown (First Word)'
    )
    df['onet_soc_code'] = 'N/A' # Default for this method
    df['onet_match_score'] = np.nan # Default for this method
    return df

@st.cache_data # Cache this potentially expensive operation
def classify_with_onet(_df_jobs, _onet_df, job_title_col='Title', min_similarity_threshold=0.2):
    """
    Classifies job titles in _df_jobs by matching them against titles in _onet_df
    using TF-IDF and cosine similarity.
    Uses 'O*NET-SOC Code', 'Title', 'Description' from _onet_df as confirmed by user.
    """
    df_jobs_classified = _df_jobs.copy()

    # Use column names as confirmed by the user for the O*NET data
    onet_title_col_name = 'Title'
    onet_code_col_name = 'O*NET-SOC Code'
    # onet_desc_col_name = 'Description' # Available if needed for future enhancements

    # These checks are now done in load_onet_data, but good to be aware
    if onet_title_col_name not in _onet_df.columns or onet_code_col_name not in _onet_df.columns:
        st.error(f"O*NET data must contain '{onet_title_col_name}' and '{onet_code_col_name}' columns.")
        df_jobs_classified['category'] = 'Error: O*NET cols missing'
        df_jobs_classified['onet_soc_code'] = 'Error'
        df_jobs_classified['onet_match_score'] = np.nan
        return df_jobs_classified
    
    onet_titles = _onet_df[onet_title_col_name].fillna('').astype(str).tolist()
    job_titles_to_classify = df_jobs_classified[job_title_col].fillna('').astype(str).tolist()

    if not onet_titles:
        st.error("O*NET data has no titles to match against.")
        df_jobs_classified['category'] = 'Error: No O*NET titles'
        df_jobs_classified['onet_soc_code'] = 'Error'
        df_jobs_classified['onet_match_score'] = np.nan
        return df_jobs_classified

    vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.9)
    
    try:
        all_titles_for_vocab = job_titles_to_classify + onet_titles
        vectorizer.fit(all_titles_for_vocab)
        job_title_vectors = vectorizer.transform(job_titles_to_classify)
        onet_title_vectors = vectorizer.transform(onet_titles)
    except ValueError as ve: 
        st.error(f"TF-IDF Vectorization error: {ve}. This can happen if titles are too short or non-descriptive.")
        df_jobs_classified['category'] = 'Error: TF-IDF failed'
        df_jobs_classified['onet_soc_code'] = 'Error'
        df_jobs_classified['onet_match_score'] = np.nan
        return df_jobs_classified

    similarity_matrix = cosine_similarity(job_title_vectors, onet_title_vectors)

    matched_onet_titles = []
    matched_onet_codes = []
    match_scores = []

    for i in range(similarity_matrix.shape[0]):
        best_match_idx = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i, best_match_idx]
        
        if best_score >= min_similarity_threshold:
            matched_onet_titles.append(_onet_df.iloc[best_match_idx][onet_title_col_name])
            matched_onet_codes.append(_onet_df.iloc[best_match_idx][onet_code_col_name])
            match_scores.append(best_score)
        else:
            matched_onet_titles.append('Unclassified (O*NET)')
            matched_onet_codes.append('N/A')
            match_scores.append(best_score) 

    df_jobs_classified['category'] = matched_onet_titles
    df_jobs_classified['onet_soc_code'] = matched_onet_codes
    df_jobs_classified['onet_match_score'] = match_scores
    
    return df_jobs_classified

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("🏷️ Job Posting Classifier & Analyzer (O*NET Aligned)")
st.markdown("Categorize ~4000 job postings using standard O*NET occupations loaded from a URL, or by simple rules.")

# --- Sidebar for Setup and Filters ---
st.sidebar.header("🛠️ Setup & Classification")

# 1. Load Main Job Data
with st.spinner("Loading main job postings data..."):
    df_loaded = load_data_from_url(DATA_URL, "Main Job Postings")

if df_loaded is None:
    st.error("Fatal Error: Could not load main job postings data. Dashboard cannot proceed.")
    st.stop()

st.sidebar.success(f"{df_loaded.shape[0]} job postings loaded.")

# 2. Select Categorization Method
categorization_method = st.sidebar.radio(
    "Choose categorization method:",
    ("O*NET Standard Classification", "First word of Title")
)

onet_df = None
df_processed = None # Initialize df_processed

if categorization_method == "O*NET Standard Classification":
    with st.spinner("Loading O*NET standard occupation data from URL..."):
        onet_df = load_onet_data_from_url(ONET_DATA_URL)
    
    if onet_df is not None:
        st.sidebar.success(f"O*NET data loaded: {onet_df.shape[0]} standard occupations.")
        if 'Title' in df_loaded.columns: # Ensure main data has 'Title' for classification
            with st.spinner("Classifying jobs using O*NET data... This might take a few moments."):
                df_processed = classify_with_onet(df_loaded, onet_df)
            st.success("O*NET classification complete!")
            # Show a small sample of matches and scores
            st.sidebar.markdown("---")
            st.sidebar.subheader("O*NET Classification Sample:")
            sample_cols = ['Title', 'category', 'onet_soc_code', 'onet_match_score']
            display_sample_cols = [col for col in sample_cols if col in df_processed.columns]
            st.sidebar.dataframe(df_processed[display_sample_cols].head(3))
            st.sidebar.metric("Average Match Score (where classified)", f"{df_processed[df_processed['category'] != 'Unclassified (O*NET)']['onet_match_score'].mean():.2f}" if not df_processed[df_processed['category'] != 'Unclassified (O*NET)'].empty else "N/A")

        else: # Main job data lacks 'Title'
            st.error("Main job data is missing the 'Title' column. Cannot perform O*NET classification.")
            df_processed = add_category_column_first_word(df_loaded) # Fallback but will show error for category
    else: # O*NET data failed to load
        st.sidebar.error("Failed to load O*NET data. O*NET classification unavailable. You can use 'First word of Title' method.")
        # Fallback to first word if O*NET fails, or let user explicitly choose
        if st.sidebar.button("Use 'First word of Title' as fallback?"):
            df_processed = add_category_column_first_word(df_loaded)
            st.info("Using 'First word of Title' for categorization due to O*NET load failure.")
        else:
            st.warning("O*NET data not available. Please select 'First word of Title' method or check O*NET URL.")
            # Create a dummy df_processed to avoid errors, but it won't be useful
            df_processed = df_loaded.copy()
            df_processed['category'] = "Classification Pending"
            df_processed['onet_soc_code'] = "N/A"
            df_processed['onet_match_score'] = np.nan


elif categorization_method == "First word of Title":
    df_processed = add_category_column_first_word(df_loaded)
    st.info("Using 'First word of Title' for categorization.")


if df_processed is not None and 'category' in df_processed.columns and not df_processed['category'].str.contains("Error|Pending", na=False).all():
    # --- Remainder of Sidebar (Filters, Display, Download) ---
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Display & Filter Options")
    # (This section can be largely copied from the previous version, ensuring it uses df_processed and df_working_set)
    unique_categories_processed = ['All'] + sorted(df_processed['category'].dropna().unique().tolist())
    selected_category_filter = st.sidebar.selectbox("Filter by Main Category:", unique_categories_processed)

    if selected_category_filter != 'All':
        df_working_set = df_processed[df_processed['category'] == selected_category_filter].copy()
    else:
        df_working_set = df_processed.copy()

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Keyword Search")
    search_term = st.sidebar.text_input("Search keyword in selected text columns:")
    available_for_search = [col for col in df_working_set.columns if df_working_set[col].dtype == 'object']
    default_search_cols = [col for col in ['Title', 'Job.Description'] if col in available_for_search] # Ensure 'Job.Description' exists
    if 'Job.Description' not in df_working_set.columns and 'Description' in df_working_set.columns: # Check for O*NET's Description
        default_search_cols = [col for col in ['Title', 'Description'] if col in available_for_search]
    if not default_search_cols and available_for_search: default_search_cols = available_for_search[:1]
    columns_to_search = st.sidebar.multiselect("Select columns for keyword search:", options=available_for_search, default=default_search_cols)

    if search_term and columns_to_search:
        search_mask = df_working_set[columns_to_search].astype(str).apply(lambda col: col.str.contains(search_term, case=False, na=False)).any(axis=1)
        df_working_set = df_working_set[search_mask]
    
    st.sidebar.markdown("---")
    st.sidebar.header("📄 Table Display")
    show_full_data_rows = st.sidebar.checkbox("Display all filtered rows in table", value=False)
    num_rows_preview = 20
    if not show_full_data_rows: num_rows_preview = st.sidebar.slider("Number of rows for preview table", 5, 200, 20, key="rows_slider")
    
    all_available_cols_for_table = df_working_set.columns.tolist()
    default_cols_for_table = ['Title', 'Position', 'Company', 'category', 'onet_soc_code', 'onet_match_score']
    actual_default_cols_for_table = [col for col in default_cols_for_table if col in all_available_cols_for_table]
    selected_cols_for_main_table = st.sidebar.multiselect("Select columns for main table display:", options=all_available_cols_for_table, default=actual_default_cols_for_table, key="table_cols_multiselect")
    if not selected_cols_for_main_table and all_available_cols_for_table:
         selected_cols_for_main_table = [col for col in ['Title', 'category'] if col in all_available_cols_for_table] or all_available_cols_for_table[:1]

    st.sidebar.markdown("---")
    st.sidebar.header("📥 Download Filtered Data")
    @st.cache_data # Cache the conversion for efficiency
    def convert_df_to_csv(df_to_convert): return df_to_convert.to_csv(index=False).encode('utf-8')
    
    # Ensure df_working_set has the selected columns before trying to download
    df_to_download = df_working_set[selected_cols_for_main_table] if selected_cols_for_main_table and all(col in df_working_set.columns for col in selected_cols_for_main_table) else df_working_set

    csv_to_download = convert_df_to_csv(df_to_download)
    st.sidebar.download_button(label="Download current data as CSV", data=csv_to_download, file_name='classified_job_postings.csv', mime='text/csv')

    # --- Main Panel Display ---
    st.header(f"📋 Displaying Job Postings ({df_working_set.shape[0]} entries match filters)")
    if selected_cols_for_main_table:
        if show_full_data_rows: st.dataframe(df_working_set[selected_cols_for_main_table], height=600)
        else: st.dataframe(df_working_set[selected_cols_for_main_table].head(num_rows_preview))
    else: st.warning("No columns selected for the main table display or no columns available.")

    st.header("📈 Analysis of Displayed Data")
    if df_working_set.empty or 'category' not in df_working_set.columns:
        st.warning("No data or 'category' column available for analysis based on current filters.")
    else:
        col_analysis1, col_analysis2 = st.columns(2)
        with col_analysis1:
            st.subheader("Main Category Distribution")
            main_category_counts_filtered = df_working_set['category'].value_counts()
            if not main_category_counts_filtered.empty:
                num_top_main_cat_chart = st.slider("Top N main categories (chart):", 1, max(1, min(30, len(main_category_counts_filtered))), min(10, max(1, len(main_category_counts_filtered))), key="main_cat_chart_slider_v2")
                st.bar_chart(main_category_counts_filtered.nlargest(num_top_main_cat_chart))
            else: st.write("No main category data for chart.")
        with col_analysis2:
            st.subheader("Value Counts (Main Categories)")
            if not main_category_counts_filtered.empty:
                num_top_main_cat_table = st.slider("Top N main categories (table):", 1, max(1, min(30, len(main_category_counts_filtered))), min(10, max(1, len(main_category_counts_filtered))), key="main_cat_table_slider_v2")
                st.dataframe(main_category_counts_filtered.nlargest(num_top_main_cat_table).reset_index().rename(columns={'index':'Category', 'category':'Count'}))
            else: st.write("No main category data for table.")
        
        st.markdown("---")
        st.subheader("Distribution of Another Feature")
        # (Secondary feature analysis - ensure this part is robust)
        potential_secondary_cols = [
            col for col in df_working_set.columns 
            if df_working_set[col].dtype == 'object' and 
                df_working_set[col].nunique() < 50 and 
                df_working_set[col].nunique() > 1 and 
                col not in ['Title', 'Job.Description', 'Description', 'category', 'Company', 'onet_soc_code']
        ]
        for known_cat_col in ['Position', 'Employment.Type', 'Industry']:
            if known_cat_col in df_working_set.columns and known_cat_col not in potential_secondary_cols:
                if df_working_set[known_cat_col].nunique() < 50 and df_working_set[known_cat_col].nunique() > 1: 
                    potential_secondary_cols.append(known_cat_col)
        
        if potential_secondary_cols:
            selected_secondary_col = st.selectbox(
                "Select feature for distribution analysis:", 
                options=potential_secondary_cols, 
                index=0 if potential_secondary_cols else -1, # Handle empty list
                key="secondary_col_selectbox"
            )
            if selected_secondary_col and selected_secondary_col in df_working_set.columns: # Ensure selection is valid
                secondary_col_counts = df_working_set[selected_secondary_col].value_counts()
                st.bar_chart(secondary_col_counts)
                if st.checkbox(f"Show value counts table for '{selected_secondary_col}'", key=f"table_for_{selected_secondary_col}_v2"):
                    st.dataframe(secondary_col_counts.reset_index().rename(columns={'index':selected_secondary_col, selected_secondary_col:'Count'}))
        else: st.write("No suitable columns for secondary distribution analysis in current filtered data.")

        # --- Canvas Visualization Section (operates on df_working_set) ---
        st.markdown("---")
        st.header("🎨 Visual Canvas of Title Categories (PCA)")
        # (Canvas visualization as before - ensure this part is robust)
        if df_working_set.empty or 'Title' not in df_working_set.columns or df_working_set['Title'].isnull().all() or len(df_working_set) < 2:
            st.warning("Not enough data or no 'Title' data in current selection for canvas visualization (need at least 2 postings).")
        elif 'category' not in df_working_set.columns or df_working_set['category'].isnull().all():
            st.warning("No 'category' data available for coloring the canvas visualization.")
        else:
            try:
                with st.spinner("Generating canvas visualization..."):
                    titles_for_plot = df_working_set['Title'].fillna('')
                    vectorizer_plot = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2)
                    tfidf_matrix_plot = vectorizer_plot.fit_transform(titles_for_plot)

                    if tfidf_matrix_plot.shape[1] < 2:
                        st.warning("Not enough distinct terms in titles to create a 2D PCA plot for current selection.")
                    else:
                        pca_plot = PCA(n_components=2, random_state=42)
                        reduced_features_plot = pca_plot.fit_transform(tfidf_matrix_plot.toarray())
                        
                        # Ensure 'category' exists and is not all NaN before trying to use it for coloring
                        if 'category' in df_working_set.columns and not df_working_set['category'].dropna().empty:
                            plot_df = pd.DataFrame({
                                'pca_x': reduced_features_plot[:,0], 
                                'pca_y': reduced_features_plot[:,1], 
                                'category': df_working_set['category'].fillna('Unknown'), # Fill NaN categories for plotting
                                'title': df_working_set['Title']
                            })
                            
                            fig_canvas, ax_canvas = plt.subplots(figsize=(12, 9))
                            unique_cats_plot_series = plot_df['category'].astype('category')
                            cat_codes = unique_cats_plot_series.cat.codes
                            num_unique_cats_plot = len(unique_cats_plot_series.cat.categories)
                            
                            cmap_canvas_name = 'tab10' if num_unique_cats_plot <= 10 else ('tab20' if num_unique_cats_plot <= 20 else 'viridis')
                            cmap_canvas = plt.get_cmap(cmap_canvas_name, num_unique_cats_plot if num_unique_cats_plot > 0 else 1)

                            scatter_canvas = ax_canvas.scatter(plot_df['pca_x'], plot_df['pca_y'], c=cat_codes, cmap=cmap_canvas, alpha=0.7, s=50, edgecolor='k', linewidths=0.5)
                            ax_canvas.set_title('Job Titles Visualized by Category (TF-IDF + PCA)', fontsize=15)
                            ax_canvas.set_xlabel('PCA Component 1', fontsize=12); ax_canvas.set_ylabel('PCA Component 2', fontsize=12)
                            ax_canvas.grid(True, linestyle='--', alpha=0.5)

                            if num_unique_cats_plot > 0:
                                legend_handles = []
                                for i, cat_name in enumerate(unique_cats_plot_series.cat.categories):
                                     # Ensure color index is within bounds for cmap
                                    color_idx = i / (num_unique_cats_plot - 1) if num_unique_cats_plot > 1 else 0.0
                                    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=str(cat_name), markerfacecolor=cmap_canvas(color_idx), markersize=8))
                                if legend_handles: ax_canvas.legend(handles=legend_handles, title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
                            
                            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
                            st.pyplot(fig_canvas)
                            if st.checkbox("Show sample titles for canvas points (first 10)?", key="show_canvas_sample_v2"): st.dataframe(plot_df[['title', 'category']].head(10))
                        else:
                            st.warning("No valid category data available to color the canvas plot.")
            except Exception as e_plot: st.error(f"Could not generate canvas: {e_plot}")
else:
    if df_loaded is not None and 'Title' not in df_loaded.columns:
        st.error("The loaded job data does not contain a 'Title' column, which is essential for all categorization methods.")
    elif df_loaded is not None and df_processed is not None and 'category' in df_processed.columns and df_processed['category'].str.contains("Error|Pending", na=False).all():
         st.error("Categorization failed or is pending. Please check the setup and data sources.")
    elif df_loaded is not None: # df_loaded exists but df_processed might be None or incomplete
        st.warning("Data processing incomplete. Please select a valid categorization method and ensure all required data (like O*NET from URL) is loaded.")


st.sidebar.markdown("---")
st.sidebar.info("Adjust options in this sidebar to filter and explore the job postings data.")
