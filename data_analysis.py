import pandas as pd
import os

def analyze_metadata(base_file_name='HAM10000_metadata.csv'):
    """
    Reads the HAM10000 metadata CSV, including duplicate copies with numbering 
    (like '(1)', '(2)', etc.), concatenates them, and performs statistical analysis.
    
    Args:
        base_file_name (str): The common name of the metadata CSV file(s).
    """
    data_frames = []
    
    # Check for the base file
    if os.path.exists(base_file_name):
        data_frames.append(pd.read_csv(base_file_name))
        print(f"Loaded: {base_file_name}")
    
    # Check for numbered duplicates
    i = 1
    while True:
        numbered_file_name = base_file_name.replace('.csv', f' ({i}).csv')
        if os.path.exists(numbered_file_name):
            data_frames.append(pd.read_csv(numbered_file_name))
            print(f"Loaded: {numbered_file_name}")
            i += 1
        else:
            break

    if not data_frames:
        print(f"Error: No CSV files found matching '{base_file_name}' or its numbered versions.")
        return

    # Concatenate all loaded dataframes
    df = pd.concat(data_frames, ignore_index=True).drop_duplicates()
    
    print(f"\n--- Combined Data Analysis ({len(df)} Unique Records) ---")
    
    # 1. Display the first few rows
    print("\n1. Head of the Dataset:")
    print(df.head())
    
    # 2. Check for missing values
    print("\n2. Missing Values Count:")
    print(df.isnull().sum())
    
    # 3. Analyze disease distribution (dx) - Including percentages
    print("\n3. Disease (dx) Distribution (Lesion Type):")
    dx_counts = df['dx'].value_counts()
    dx_percent = df['dx'].value_counts(normalize=True).mul(100).round(2)
    dx_report = pd.DataFrame({'Count': dx_counts, 'Percentage': dx_percent})
    dx_report['Percentage'] = dx_report['Percentage'].astype(str) + '%'
    print(dx_report)
    
    # 4. Analyze age distribution
    print("\n4. Age Statistics (in Years):")
    print(df['age'].describe().to_string())
    
    # 5. Analyze gender distribution - Including percentages
    print("\n5. Sex Distribution:")
    sex_counts = df['sex'].value_counts()
    sex_percent = df['sex'].value_counts(normalize=True).mul(100).round(2)
    sex_report = pd.DataFrame({'Count': sex_counts, 'Percentage': sex_percent})
    sex_report['Percentage'] = sex_report['Percentage'].astype(str) + '%'
    print(sex_report)

    # 6. Localization (Body Part) Distribution
    print("\n6. Localization Distribution (Body Part):")
    loc_counts = df['localization'].value_counts()
    loc_percent = df['localization'].value_counts(normalize=True).mul(100).round(2)
    loc_report = pd.DataFrame({'Count': loc_counts, 'Percentage': loc_percent})
    loc_report['Percentage'] = loc_report['Percentage'].astype(str) + '%'
    print(loc_report)
    
    print("\nTo perform visual analysis (like histograms/bar charts), you would typically install matplotlib and seaborn.")

# Run the analysis
# This function will automatically look for 'HAM10000_metadata.csv', 'HAM10000_metadata (1).csv', etc.
analyze_metadata()
