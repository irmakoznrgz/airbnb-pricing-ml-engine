import streamlit as st
import pandas as pd
import joblib
import altair as alt
import warnings

warnings.filterwarnings("ignore")

logo_url = "https://s3-symbol-logo.tradingview.com/airbnb--big.svg"

st.set_page_config(
    page_title="Airbnb Istanbul Price Prediction",
    page_icon=logo_url,
    layout="wide"
)
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #FF5A5F;
        color: white;
        border-radius: 8px;
        height: 50px;
        width: 100%;
        font-size: 20px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #E0484D;
        color: white;
        box-shadow: 0px 4px 10px rgba(224, 72, 77, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_columns():
    model = joblib.load("models/best_lgbm_model.pkl")
    df_template = pd.read_csv("data/processed/cleaned_listings.csv", nrows=1)
    model_columns = df_template.drop(columns=['price']).columns
    return model, model_columns

model, expected_columns = load_model_and_columns()

sparkline_data = pd.DataFrame({
    "x": range(40),
    "y": [1, 2, 3, 4, 6, 8, 12, 15, 20, 25, 22, 18, 14, 10, 8, 6, 5, 4, 5, 8, 15, 25, 35, 45, 40, 30, 20, 15, 10, 8, 6, 4, 3, 2, 2, 1, 1, 1, 0, 0]
})
mini_chart = alt.Chart(sparkline_data).mark_bar(
    size=12, 
    color='#FF5A5F', 
    opacity=0.3, 
    cornerRadiusTopLeft=3, 
    cornerRadiusTopRight=3
).encode(
    x=alt.X("x", axis=None), 
    y=alt.Y("y", axis=None), 
    tooltip=alt.value(None)  
).properties(
    height=60 
).configure_view(strokeWidth=0)
st.altair_chart(mini_chart, use_container_width=True)


st.markdown(
    f'<h1 style="white-space: nowrap;"><img src="{logo_url}" width="55" style="vertical-align: middle; margin-right: 15px; padding-bottom: 5px;" alt="Airbnb Logo">Istanbul Airbnb Price Predictor</h1>', 
    unsafe_allow_html=True
)  
st.markdown("Enter your property details and let our AI-powered model predict the ideal nightly price.")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Location & Type")
    district_list = [col.replace("neighbourhood_cleansed_", "").replace("_", " ") 
                 for col in expected_columns if col.startswith("neighbourhood_cleansed_")]
    neighbourhood = st.selectbox("Where", district_list)

    property_list = [col.replace("property_type_", "").replace("_", " ") 
                 for col in expected_columns if col.startswith("property_type_")]
    property_type = st.selectbox("Property Type", property_list)


    room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])

with col2:
    st.subheader("Capacity & Rooms")
    accommodates = st.slider("Who", 1, 16, 2)
    bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=10.0, value=1.0, step=1.0)
    beds = st.number_input("Beds", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

with col3:
    st.subheader("Amenities & Features")
    has_wifi = st.checkbox("Wi-Fi", value=True)
    has_ac = st.checkbox("Air Conditioning")
    has_view = st.checkbox("View (Sea/Bosphorus)")
    has_elevator = st.checkbox("Elevator")
    has_parking = st.checkbox("Parking")

st.markdown("---")

if st.button("Predict Price"):
    user_data = {
        'accommodates': accommodates,
        'bedrooms': bedrooms,
        'beds': beds,
        'bathrooms': bathrooms,
        'has_wifi': int(has_wifi),
        'has_ac': int(has_ac),
        'has_view': int(has_view),
        'has_elevator': int(has_elevator),
        'has_parking': int(has_parking),
        'neighbourhood_cleansed': neighbourhood,
        'property_type': property_type,
        'room_type': room_type,
        'minimum_nights': 2,
        'number_of_reviews': 10,
        'review_scores_rating': 4.8,
        'host_is_superhost': 0,
        'latitude': 41.0082,
        'longitude': 28.9784
    }
    
    user_df = pd.DataFrame([user_data])
    user_df = pd.get_dummies(user_df)
    user_df.columns = user_df.columns.str.replace(r'\W+', '_', regex=True)
    
    final_features = user_df.reindex(columns=expected_columns, fill_value=0)
    final_features = final_features.astype(float)
    
    prediction = model.predict(final_features)[0]
    
    CITY_AVG = 2200.0 
    difference = prediction - CITY_AVG
    diff_percent = (abs(difference) / CITY_AVG) * 100
    
    st.subheader("Prediction Results")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(
            label="Suggested Nightly Price",
            value=f"{prediction:,.2f} TL",
            delta=f"{difference:+,.2f} TL vs City Average"
        )
        
    with res_col2:
        if difference > 0:
            st.info(f"**Premium Valuation:** Your property is valued **{diff_percent:.1f}% higher** than the average Istanbul listing ({CITY_AVG:,.0f} TL). Premium features or a central location justify this competitive advantage!")
        else:
            st.success(f"**Competitive Pricing:** Your property is priced **{diff_percent:.1f}% below** the city average ({CITY_AVG:,.0f} TL). This is a highly attractive rate that can secure higher booking volumes and occupancy rates!")

    st.balloons()