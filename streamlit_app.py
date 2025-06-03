import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans # Though KMeans isn't used for coloring here, it was in prev thoughts
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np # For handling potential NaN in PCA results if input is all zeros

DATA_URL = 'https://raw.githubusercontent.com/adinplb/largedataset-JRec/refs/heads/main/Filtered_Jobs_4000.csv'

@st.cache_data
def load_data_from_url(url):
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")
        return None

def add_category_column(df_input):
    df = df_input.copy()
    if 'Title' not in df.columns:
        st.error("Error: 'Title' column not found. Cannot create categories.")
        return df
    df['category'] = df['Title'].apply(
        lambda x: x.split()[0] if isinstance(x, str) and x.strip() else 'Unknown'
    )
    return df

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("📊 Enhanced Job Postings Analysis Dashboard")
st.markdown("Categorizing jobs and allowing deeper exploration with filters, search, and visualizations.")

df_loaded = load_data_from_url(DATA_URL)

if df_loaded is not None:
    st.success(f"Successfully loaded {df_loaded.shape[0]} job postings.")
    df_processed = add_category_column(df_loaded)

    if 'category' in df_processed.columns:
        # --- Sidebar for Controls ---
        st.sidebar.header("⚙️ Display & Filter Options")
        unique_categories = ['All'] + sorted(df_processed['category'].dropna().unique().tolist())
        selected_category_filter = st.sidebar.selectbox("Filter by Main Category:", unique_categories)

        if selected_category_filter != 'All':
            df_working_set = df_processed[df_processed['category'] == selected_category_filter].copy()
        else:
            df_working_set = df_processed.copy()

        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Keyword Search")
        search_term = st.sidebar.text_input("Search keyword in selected text columns:")
        available_for_search = [col for col in df_working_set.columns if df_working_set[col].dtype == 'object']
        default_search_cols = [col for col in ['Title', 'Job.Description'] if col in available_for_search]
        if not default_search_cols and available_for_search:
            default_search_cols = available_for_search[:1]
        columns_to_search = st.sidebar.multiselect("Select columns for keyword search:", options=available_for_search, default=default_search_cols)

        if search_term and columns_to_search:
            search_mask = df_working_set[columns_to_search].astype(str).apply(lambda col: col.str.contains(search_term, case=False, na=False)).any(axis=1)
            df_working_set = df_working_set[search_mask]
        
        st.sidebar.markdown("---")
        st.sidebar.header("📄 Table Display")
        show_full_data_rows = st.sidebar.checkbox("Display all filtered rows in table", value=False)
        num_rows_preview = 20
        if not show_full_data_rows:
            num_rows_preview = st.sidebar.slider("Number of rows for preview table", 5, 200, 20)
        
        all_available_cols_for_table = df_working_set.columns.tolist()
        default_cols_for_table = ['Title', 'Position', 'Company', 'Industry', 'category']
        actual_default_cols_for_table = [col for col in default_cols_for_table if col in all_available_cols_for_table]
        selected_cols_for_main_table = st.sidebar.multiselect("Select columns for main table display:", options=all_available_cols_for_table, default=actual_default_cols_for_table)
        if not selected_cols_for_main_table and all_available_cols_for_table:
             selected_cols_for_main_table = [col for col in ['Title', 'category'] if col in all_available_cols_for_table] or all_available_cols_for_table[:1]

        st.sidebar.markdown("---")
        st.sidebar.header("📥 Download Filtered Data")
        @st.cache_data
        def convert_df_to_csv(df_to_convert):
            return df_to_convert.to_csv(index=False).encode('utf-8')
        csv_to_download = convert_df_to_csv(df_working_set[selected_cols_for_main_table] if selected_cols_for_main_table else df_working_set)
        st.sidebar.download_button(label="Download current data as CSV", data=csv_to_download, file_name='filtered_job_postings.csv', mime='text/csv')

        # --- Main Panel Display ---
        st.header(f"📋 Displaying Job Postings ({df_working_set.shape[0]} entries match filters)")
        if selected_cols_for_main_table:
            if show_full_data_rows:
                st.dataframe(df_working_set[selected_cols_for_main_table], height=600)
            else:
                st.dataframe(df_working_set[selected_cols_for_main_table].head(num_rows_preview))
        else:
            st.warning("No columns selected for the main table display or no columns available.")

        st.header("📈 Analysis of Displayed Data")
        if df_working_set.empty:
            st.warning("No data available for analysis based on current filters.")
        else:
            col_analysis1, col_analysis2 = st.columns(2)
            with col_analysis1:
                st.subheader("Main Category Distribution")
                main_category_counts_filtered = df_working_set['category'].value_counts()
                num_top_main_cat_chart = st.slider("Number of top main categories for chart:", 1, max(1, min(30, len(main_category_counts_filtered))), min(10, max(1, len(main_category_counts_filtered))), key="main_cat_chart_slider")
                if not main_category_counts_filtered.empty:
                    st.bar_chart(main_category_counts_filtered.nlargest(num_top_main_cat_chart))
                else:
                    st.write("No main category data for chart.")
            with col_analysis2:
                st.subheader("Value Counts (Main Categories)")
                num_top_main_cat_table = st.slider("Number of top main categories for table:", 1, max(1, min(30, len(main_category_counts_filtered))), min(10, max(1, len(main_category_counts_filtered))), key="main_cat_table_slider")
                if not main_category_counts_filtered.empty:
                    st.dataframe(main_category_counts_filtered.nlargest(num_top_main_cat_table).reset_index().rename(columns={'index':'Category', 'category':'Count'}))
                else:
                    st.write("No main category data for table.")
            
            st.markdown("---")
            st.subheader("Distribution of Another Feature")
            potential_secondary_cols = [col for col in df_working_set.columns if df_working_set[col].dtype == 'object' and df_working_set[col].nunique() < 50 and df_working_set[col].nunique() > 1 and col not in ['Title', 'Job.Description', 'category', 'Company']]
            for known_cat_col in ['Position', 'Employment.Type', 'Industry']: # Ensure these common ones are available if they exist
                if known_cat_col in df_working_set.columns and known_cat_col not in potential_secondary_cols:
                    if df_working_set[known_cat_col].nunique() < 50 and df_working_set[known_cat_col].nunique() > 1:
                         potential_secondary_cols.append(known_cat_col)
            
            if potential_secondary_cols:
                selected_secondary_col = st.selectbox("Select a feature to see its distribution (within current filters):", options=potential_secondary_cols, index=0 if potential_secondary_cols else -1)
                if selected_secondary_col:
                    secondary_col_counts = df_working_set[selected_secondary_col].value_counts()
                    st.bar_chart(secondary_col_counts)
                    if st.checkbox(f"Show value counts table for '{selected_secondary_col}'", key=f"table_for_{selected_secondary_col}"):
                        st.dataframe(secondary_col_counts.reset_index().rename(columns={'index':selected_secondary_col, selected_secondary_col:'Count'}))
            else:
                st.write("No suitable categorical columns found for this analysis in the current filtered data.")

        # --- NEW: Canvas Visualization Section ---
        st.markdown("---")
        st.header("🎨 Visual Canvas of Title Categories (PCA)")
        st.markdown("This plot visualizes job titles in a 2D space based on their TF-IDF features, colored by their 'first-word' category. It helps to see if titles starting with the same word tend to cluster.")

        if df_working_set.empty or 'Title' not in df_working_set.columns or df_working_set['Title'].isnull().all():
            st.warning("Not enough data or no 'Title' data in the current selection to generate the canvas visualization.")
        elif len(df_working_set) < 2 : # PCA needs at least 2 samples
            st.warning(f"Only {len(df_working_set)} job posting(s) in the current selection. Need at least 2 to generate PCA plot.")
        else:
            try:
                with st.spinner("Generating canvas visualization..."):
                    # 1. TF-IDF Vectorization on 'Title'
                    titles_for_plot = df_working_set['Title'].fillna('')
                    vectorizer_plot = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2) # Adjusted min_df for robustness
                    tfidf_matrix_plot = vectorizer_plot.fit_transform(titles_for_plot)

                    if tfidf_matrix_plot.shape[1] < 2: # PCA needs at least 2 features
                        st.warning("Not enough distinct terms in titles to create a meaningful 2D PCA plot for the current selection.")
                    else:
                        # 2. Dimensionality Reduction (PCA)
                        pca_plot = PCA(n_components=2, random_state=42)
                        reduced_features_plot = pca_plot.fit_transform(tfidf_matrix_plot.toarray())

                        plot_df = pd.DataFrame({
                            'pca_x': reduced_features_plot[:, 0],
                            'pca_y': reduced_features_plot[:, 1],
                            'category': df_working_set['category'], # Use existing 'first-word' category
                            'title': df_working_set['Title']
                        })
                        
                        # Map string categories to numbers for coloring if not done automatically by cmap
                        unique_cats_plot = plot_df['category'].astype('category') # Convert to categorical type
                        cat_codes = unique_cats_plot.cat.codes


                        # 3. Create Scatter Plot
                        fig_canvas, ax_canvas = plt.subplots(figsize=(12, 9))
                        
                        # Use a suitable colormap
                        # Ensure there are enough colors for the number of unique categories
                        num_unique_cats_plot = len(unique_cats_plot.cat.categories)
                        cmap_canvas_name = 'tab10' if num_unique_cats_plot <= 10 else ('tab20' if num_unique_cats_plot <=20 else 'viridis')
                        cmap_canvas = plt.get_cmap(cmap_canvas_name, num_unique_cats_plot)

                        scatter_canvas = ax_canvas.scatter(
                            plot_df['pca_x'], plot_df['pca_y'], 
                            c=cat_codes, # Use integer codes for categories
                            cmap=cmap_canvas, 
                            alpha=0.7, s=50, edgecolor='k',linewidths=0.5
                        )

                        ax_canvas.set_title('Job Titles Visualized by Category (TF-IDF + PCA)', fontsize=15)
                        ax_canvas.set_xlabel('PCA Component 1', fontsize=12)
                        ax_canvas.set_ylabel('PCA Component 2', fontsize=12)
                        ax_canvas.grid(True, linestyle='--', alpha=0.5)

                        # Create a legend
                        legend_handles = []
                        for i, cat_name in enumerate(unique_cats_plot.cat.categories):
                            legend_handles.append(
                                plt.Line2D([0], [0], marker='o', color='w', 
                                           label=cat_name,
                                           markerfacecolor=cmap_canvas(i / (num_unique_cats_plot -1 if num_unique_cats_plot > 1 else 1) ), # Normalize index for cmap
                                           markersize=8)
                            )
                        if legend_handles: # Add legend only if there are categories
                             ax_canvas.legend(handles=legend_handles, title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
                        
                        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend outside

                        st.pyplot(fig_canvas)
                        
                        if st.checkbox("Show sample titles for canvas points (first 10 with categories)?", key="show_sample_titles_canvas"):
                            st.dataframe(plot_df[['title', 'category']].head(10))

            except Exception as e_plot:
                st.error(f"Could not generate canvas visualization: {e_plot}")
    else:
        st.warning("The 'category' column could not be generated. Cannot proceed with dashboard.")
else:
    st.error("Fatal Error: Could not load job postings data. Please check the data source URL or your network connection.")

st.sidebar.markdown("---")
st.sidebar.info("Dashboard for job posting analysis.")
