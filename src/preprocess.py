import pandas as pd

def clean_target_variable(df):
    df = df.copy()
    df['price'] = df['price'].replace({r'\$': '', ',': ''}, regex=True).astype(float)
    df = df.dropna(subset=['price'])
    return df

def select_features(df):
    cols_to_keep = [
        'neighbourhood_cleansed',
        'latitude', 
        'longitude',
        'property_type', 
        'room_type',
        'accommodates',
        'bathrooms_text', 
        'bedrooms', 
        'beds',
        'amenities',
        'minimum_nights', 
        'maximum_nights',
        'availability_365',
        'number_of_reviews', 
        'review_scores_rating',
        'host_is_superhost',
        'host_identity_verified',
        'calculated_host_listings_count',
        'instant_bookable',
        'reviews_per_month',
        'review_scores_location',
        'review_scores_cleanliness',
        'price'
    ]

    existing_cols = [col for col in cols_to_keep if col in df.columns]
    return df[existing_cols]

def handle_missing_values(df):
    df = df.copy()
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    for col in ['review_scores_rating', 'review_scores_location', 'review_scores_cleanliness']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    if 'bedrooms' in df.columns and 'room_type' in df.columns:
        df['bedrooms'] = df['bedrooms'].fillna(df.groupby('room_type')['bedrooms'].transform('median'))
    if 'beds' in df.columns and 'accommodates' in df.columns:
        df['beds'] = df['beds'].fillna(df.groupby('accommodates')['beds'].transform('median'))
        
    if 'bathrooms_text' in df.columns:
        df['bathrooms_text'] = df['bathrooms_text'].fillna(df['bathrooms_text'].mode()[0])
    df['minimum_nights'] = df['minimum_nights'].fillna(df['minimum_nights'].median())
    df['maximum_nights'] = df['maximum_nights'].fillna(df['maximum_nights'].median())
    
    return df


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    if lower_bound < 0:
        lower_bound = 0
        
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

def feature_engineering(df):
    df = df.copy()
    if 'bathrooms_text' in df.columns:
        df['bathrooms'] = df['bathrooms_text'].str.extract('([0-9.]+)', expand=False).astype(float)
        df['bathrooms'] = df['bathrooms'].fillna(0.5)
        df = df.drop(columns=['bathrooms_text'])
        
    tf_columns = ['host_is_superhost', 'host_identity_verified']
    for col in tf_columns:
        if col in df.columns:
            df[col] = df[col].map({'t': 1, 'f': 0, 'True': 1, 'False': 0})
            df[col] = df[col].fillna(0)
            
    if 'amenities' in df.columns:
        df['has_wifi'] = df['amenities'].str.contains('Wifi|Internet', case=False, na=False).astype(int)
        df['has_ac'] = df['amenities'].str.contains('Air conditioning', case=False, na=False).astype(int)
        df['has_pool'] = df['amenities'].str.contains('Pool', case=False, na=False).astype(int)
        df['has_parking'] = df['amenities'].str.contains('parking|garage', case=False, na=False).astype(int)
        df['has_view'] = df['amenities'].str.contains('view|sea|bosphorus', case=False, na=False).astype(int)
        df['has_elevator'] = df['amenities'].str.contains('elevator', case=False, na=False).astype(int)
        df['is_pet_friendly'] = df['amenities'].str.contains('pets allowed', case=False, na=False).astype(int)
        df['has_balcony'] = df['amenities'].str.contains('balcony|terrace|patio', case=False, na=False).astype(int)
        df = df.drop(columns=['amenities'])
        
    return df

def encode_categoricals(df):
    df = df.copy()
    if 'property_type' in df.columns:
        type_counts = df['property_type'].value_counts()
        rare_types = type_counts[type_counts < 100].index
        df['property_type'] = df['property_type'].replace(rare_types, 'Other')
        
    if 'neighbourhood_cleansed' in df.columns:
        nhood_counts = df['neighbourhood_cleansed'].value_counts()
        rare_nhoods = nhood_counts[nhood_counts < 200].index
        df['neighbourhood_cleansed'] = df['neighbourhood_cleansed'].replace(rare_nhoods, 'Other')
        
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        
    df = df.fillna(0)
    
    return df

def run_preprocessing_pipeline(df):
    print("Data preprocessing is starting...")
    
    df = clean_target_variable(df)
    df = select_features(df)
    df = handle_missing_values(df)
    
    df = remove_outliers_iqr(df, 'price')
    df = remove_outliers_iqr(df, 'minimum_nights')
    
    df = feature_engineering(df)
    df = encode_categoricals(df)
    
    print(f"Preprocessing completed. Final Shape: {df.shape}")
    return df