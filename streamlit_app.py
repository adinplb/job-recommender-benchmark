import streamlit as st
import pandas as pd
import re # For basic keyword extraction

# --- Constants ---
# DATA_URL = 'https://raw.githubusercontent.com/adinplb/largedataset-JRec/refs/heads/main/Filtered_Jobs_4000.csv'
# In a real scenario, you would load your data here.
# For now, we'll use a sample DataFrame.

FEATURES_TO_COMBINE = [
    'Status', 'Title', 'Position', 'Company',
    'City', 'State.Name', 'Industry', 'Job.Description',
    'Employment.Type', 'Education.Required'
]
JOB_DETAIL_FEATURES_TO_DISPLAY = [
    'Company', 'Status', 'City', 'Job.Description', 'Employment.Type',
    'Position', 'Industry', 'Education.Required', 'State.Name', 'Title' # Added Title for clarity
]

# --- Helper Functions ---
@st.cache_data # Cache the data loading to improve performance
def load_data():
    """
    Loads the data.
    Replace this with your actual data loading logic, e.g., pd.read_csv(DATA_URL)
    """
    data = {
        'Status': ['Full-time', 'Part-time', 'Full-time', 'Full-time', 'Contract', 'Full-time', 'Full-time', 'Part-time', 'Full-time', 'Full-time'],
        'Title': [
            'Software Engineer', 'Data Analyst', 'Senior Software Engineer', 'Product Manager',
            'UX Designer', 'Software Engineer', 'Marketing Manager', 'Junior Data Analyst',
            'Senior Product Manager', 'DevOps Engineer'
        ],
        'Position': [
            'Mid-Level', 'Entry-Level', 'Senior-Level', 'Manager',
            'Mid-Level', 'Mid-Level', 'Manager', 'Entry-Level',
            'Senior-Level', 'Mid-Level'
        ],
        'Company': ['Tech Solutions Inc.', 'Data Insights LLC', 'Innovate Corp', 'Productive Co.',
                    'Creative Designs', 'Tech Solutions Inc.', 'Market Growth Ltd.', 'Data Insights LLC',
                    'Productive Co.', 'Cloud Services Co.'],
        'City': ['San Francisco', 'New York', 'San Francisco', 'Austin',
                 'Remote', 'Boston', 'Chicago', 'New York',
                 'Austin', 'Seattle'],
        'State.Name': ['California', 'New York', 'California', 'Texas',
                       'N/A', 'Massachusetts', 'Illinois', 'New York',
                       'Texas', 'Washington'],
        'Industry': ['Technology', 'Analytics', 'Technology', 'Software',
                     'Design', 'Technology', 'Marketing', 'Analytics',
                     'Software', 'Technology'],
        'Job.Description': [
            'Develop and maintain web applications.', 'Analyze data to provide insights.',
            'Lead development of new software features.', 'Define product strategy and roadmap.',
            'Design user-friendly interfaces for web and mobile.', 'Build scalable software solutions.',
            'Develop and execute marketing campaigns.', 'Assist senior analysts with data tasks.',
            'Oversee product lifecycle from conception to launch.', 'Manage and automate cloud infrastructure.'
        ],
        'Employment.Type': ['Permanent', 'Permanent', 'Permanent', 'Permanent',
                            'Contract', 'Permanent', 'Permanent', 'Permanent',
                            'Permanent', 'Permanent'],
        'Education.Required': [
            "Bachelor's Degree in Computer Science", "Bachelor's Degree in Statistics or related",
            "Master's Degree in Computer Science", "Bachelor's or Master's Degree",
            "Bachelor's Degree in Design", "Bachelor's Degree in Computer Science",
            "Bachelor's Degree in Marketing", "Associate's or Bachelor's Degree",
            "MBA or equivalent experience", "Bachelor's Degree in IT or related"
        ]
    }
    df = pd.DataFrame(data)
    # Ensure all required columns exist, fill with NA if not (important for real data)
    for col in FEATURES_TO_COMBINE:
        if col not in df.columns:
            df[col] = pd.NA
    return df

def get_broad_categories(titles_series):
    """
    Creates broader categories from job titles using keywords.
    This is a very basic example. You might want a more sophisticated approach.
    """
    categories = {}
    keyword_map = {
        'Engineer': 'Engineering',
        'Analyst': 'Analytics',
        'Manager': 'Management',
        'Designer': 'Design',
        'Developer': 'Development', # Could be grouped with Engineering
        'Scientist': 'Research & Science'
    }
    # Ensure titles_series is Series of strings and handle NaN
    titles_series = titles_series.astype(str).fillna('')

    for title in titles_series.unique():
        assigned_category = 'Other' # Default category
        for keyword, category_name in keyword_map.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', title, re.IGNORECASE):
                assigned_category = category_name
                break
        if assigned_category not in categories:
            categories[assigned_category] = []
        categories[assigned_category].append(title)
    return categories

# --- Main Application ---
def main():
    st.set_page_config(layout="wide", page_title="Job Listings Dashboard")

    st.title("📄 Job Listings Dashboard")
    st.markdown("Explore job listings categorized by title.")

    # Load data
    df = load_data()

    if df.empty:
        st.error("No data loaded. Please check the data source.")
        return

    # --- Sidebar for Filters ---
    st.sidebar.header("Filters & Categories")

    # Option to categorize by exact title or broader category
    categorization_method = st.sidebar.radio(
        "Categorize by:",
        ("Exact Job Title", "Broad Category (Experimental)")
    )

    selected_category = None

    if 'Title' not in df.columns:
        st.error("The 'Title' column is missing from the dataset, which is required for categorization.")
        return

    if categorization_method == "Exact Job Title":
        # Get unique job titles for categorization
        unique_titles = sorted(df['Title'].astype(str).fillna('Unknown Title').unique())
        if not unique_titles:
            st.sidebar.warning("No job titles found to categorize.")
            return
        selected_category = st.sidebar.selectbox("Select Job Title Category:", ["All Jobs"] + unique_titles)
    else: # Broad Category
        broad_categories_map = get_broad_categories(df['Title'])
        if not broad_categories_map:
            st.sidebar.warning("Could not generate broad categories.")
            return

        broad_category_names = ["All Jobs"] + sorted(broad_categories_map.keys())
        selected_broad_category = st.sidebar.selectbox("Select Broad Category:", broad_category_names)

        if selected_broad_category != "All Jobs":
            # Filter titles within the selected broad category for a second-level selection
            titles_in_category = sorted(broad_categories_map[selected_broad_category])
            if titles_in_category:
                selected_category = st.sidebar.selectbox(f"Select Specific Title in '{selected_broad_category}':", ["All in Category"] + titles_in_category)
                if selected_category == "All in Category":
                    # This means we filter by the broad category's titles
                    pass # Handled in filtering logic below
            else:
                st.sidebar.info(f"No specific titles found for category '{selected_broad_category}'.")
                selected_category = "All in Category" # effectively filtering by the broad category
        else:
            selected_category = "All Jobs"


    # --- Main Panel for Displaying Jobs ---
    st.header("Job Listings")

    # Filter data based on selection
    if selected_category == "All Jobs" or selected_category is None :
        filtered_df = df
    elif categorization_method == "Exact Job Title":
        filtered_df = df[df['Title'] == selected_category]
    elif categorization_method == "Broad Category (Experimental)":
        if selected_broad_category != "All Jobs":
            titles_to_filter = broad_categories_map.get(selected_broad_category, [])
            if selected_category != "All in Category" and selected_category in titles_to_filter:
                 # User selected a specific title within a broad category
                filtered_df = df[df['Title'] == selected_category]
            else:
                # User selected "All in Category" or the specific title list was empty
                filtered_df = df[df['Title'].isin(titles_to_filter)]
        else: # Should not happen if "All Jobs" is handled above, but as a fallback
            filtered_df = df


    if filtered_df.empty:
        st.info("No jobs found for the selected category.")
    else:
        st.write(f"Displaying {len(filtered_df)} job(s):")

        # Ensure all display columns exist
        display_cols_present = [col for col in JOB_DETAIL_FEATURES_TO_DISPLAY if col in filtered_df.columns]
        missing_display_cols = [col for col in JOB_DETAIL_FEATURES_TO_DISPLAY if col not in filtered_df.columns]
        if missing_display_cols:
            st.warning(f"Note: The following detail columns are missing from the data and cannot be displayed: {', '.join(missing_display_cols)}")


        if not display_cols_present:
            st.error("None of the specified job detail features are available in the data.")
            return

        # Display job listings
        for index, row in filtered_df.iterrows():
            # Use Title as the expander label, or Company if Title is missing
            expander_label = row.get('Title', row.get('Company', f"Job {index + 1}"))
            if pd.isna(expander_label) or expander_label == '':
                expander_label = f"Job {index + 1}"

            with st.expander(f"**{expander_label}** at {row.get('Company', 'N/A')}"):
                for feature in display_cols_present:
                    # Ensure feature value is not NaN before displaying
                    value = row[feature]
                    if pd.notna(value):
                        st.markdown(f"**{feature.replace('.', ' ')}:** {value}")
                    else:
                        st.markdown(f"**{feature.replace('.', ' ')}:** Not Available")
                st.markdown("---")

    st.sidebar.markdown("---")
    st.sidebar.info("This dashboard helps visualize job data. Replace the sample data with your actual dataset.")

if __name__ == '__main__':
    main()
