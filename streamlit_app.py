import pandas as pd

DATA_URL = 'https://raw.githubusercontent.com/adinplb/largedataset-JRec/refs/heads/main/Filtered_Jobs_4000.csv'

def load_data_from_url(url):
    """
    Loads data from a CSV file URL.
    """
    try:
        df = pd.read_csv(url)
        print(f"Data loaded successfully from {url}")
        print(f"Shape of the dataframe: {df.shape}")
        # print("Column names:", df.columns.tolist()) # Uncomment to see all columns
        return df
    except Exception as e:
        print(f"Error loading data from {url}: {e}")
        return None

def main():
    df = load_data_from_url(DATA_URL)

    if df is None:
        print("Could not proceed due to data loading issues.")
        return

    # Verify 'Title' column exists, as our categorization logic depends on it.
    # Based on the filename and your description, 'Title' should be one of the columns.
    if 'Title' not in df.columns:
        print("Error: 'Title' column not found in the loaded data.")
        print("Please check the column names in the CSV file.")
        print("Available columns are:", df.columns.tolist())
        return

    # Create the 'category' column by taking the first word of the 'Title'.
    # Handles cases where 'Title' might be NaN, not a string, or an empty string.
    df['category'] = df['Title'].apply(
        lambda x: x.split()[0] if isinstance(x, str) and x.strip() else 'Unknown'
    )

    print("\nJob Postings with new 'category' column (first 20 rows):\n")
    
    # Define which columns to display for a concise overview.
    # We'll include 'Title', a few other relevant fields, and the new 'category'.
    columns_to_display = ['Title', 'Position', 'Company', 'Industry', 'category']
    
    # Ensure we only try to display columns that actually exist in the loaded DataFrame.
    existing_columns_to_display = [col for col in columns_to_display if col in df.columns]
    
    if not existing_columns_to_display:
        # Fallback if none of the preferred columns are found (unlikely if 'Title' exists)
        print("Warning: None of the preferred columns for display exist.")
        if 'Title' in df.columns and 'category' in df.columns: # Should always be true if we reach here
             print(df[['Title', 'category']].head(20).to_string())
        else:
            print("Cannot display the requested table.")
        return

    print(df[existing_columns_to_display].head(20).to_string())
    
    # Optionally, show the distribution of the top categories
    print(f"\n\nValue counts for the new 'category' column (top 10 most frequent):")
    try:
        print(df['category'].value_counts().nlargest(10))
    except Exception as e:
        print(f"Could not display value counts for category: {e}")


if __name__ == '__main__':
    main()
