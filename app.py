
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------
# Load and prepare data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Laptop_price.csv")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


@st.cache_resource
def train_model(df):
    df = df.copy()

    le_brand = LabelEncoder()
    df["Brand"] = le_brand.fit_transform(df["Brand"])

    feature_cols = ["Brand", "Processor_Speed", "RAM_Size",
                    "Storage_Capacity", "Screen_Size", "Weight"]

    X = df[feature_cols]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    return model, le_brand, feature_cols, metrics


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Laptop Price Predictor", page_icon="", layout="wide")

    # ========== SIDE BANNER ==========
    st.sidebar.image(
        "https://images.unsplash.com/photo-1518770660439-4636190af475",
        use_container_width=True
    )
    st.sidebar.markdown("###  Laptop Price Predictor")
    st.sidebar.write("Adjust the specs and see the predicted price in real-time.")

    # Team Members section
    with st.sidebar.expander("Team Members"):
        st.write("- Devaj TN")
        st.write("- Sahil Shahanas")
        st.write("- Razik Rahman M S")
        st.write("- George Pramod Thomas")
        st.write("- Abhivav K")

    # ========== TOP IMAGES ==========
    col_banner1, col_banner2 = st.columns([2, 1])

    with col_banner1:
        st.image(
            "https://images.unsplash.com/photo-1517336714731-489689fd1ca8",
            use_container_width=True,
            caption="Work smarter with data-driven pricing "
        )

    with col_banner2:
        st.image(
            "https://images.unsplash.com/photo-1518770660439-4636190af475",
            use_container_width=True,
            caption="Machine Learning in action "
        )

    st.title("Laptop Price Prediction App")
    st.write(
        "This app uses a **Linear Regression** model trained on your dataset "
        "`Laptop_price.csv` to predict laptop prices based on specifications."
    )

    df = load_data()
    model, le_brand, feature_cols, metrics = train_model(df)

    # ========== METRICS & IMAGE ==========
    mcol1, mcol2 = st.columns([2, 1])

    with mcol1:
        st.subheader(" Model Performance")
        st.write(f"**R² Score:** {metrics['r2']:.4f}")
        st.write(f"**MAE:** {metrics['mae']:.2f}")
        st.write(f"**RMSE:** {metrics['rmse']:.2f}")

    with mcol2:
        st.image(
            "https://images.unsplash.com/photo-1587613864521-9ef8dfe617cc",
            use_container_width=True,
            caption="Predict before you buy "
        )

    st.markdown("---")
    st.subheader(" Enter Laptop Specifications")

    # Use original brands for dropdown
    raw_df = pd.read_csv("Laptop_price.csv")
    brand_options = sorted(raw_df["Brand"].dropna().unique())

    # Ranges from dataset
    processor_min = float(df["Processor_Speed"].min())
    processor_max = float(df["Processor_Speed"].max())

    ram_min = int(df["RAM_Size"].min())
    ram_max = int(df["RAM_Size"].max())

    storage_min = int(df["Storage_Capacity"].min())
    storage_max = int(df["Storage_Capacity"].max())

    screen_min = float(df["Screen_Size"].min())
    screen_max = float(df["Screen_Size"].max())

    weight_min = float(df["Weight"].min())
    weight_max = float(df["Weight"].max())

    col1, col2 = st.columns(2)

    with col1:
        brand_input = st.selectbox("Brand", brand_options)
        processor_speed = st.slider(
            "Processor Speed",
            min_value=round(processor_min, 2),
            max_value=round(processor_max, 2),
            value=round((processor_min + processor_max) / 2, 2),
            step=0.01
        )
        ram_size = st.slider(
            "RAM Size (GB)",
            min_value=ram_min,
            max_value=ram_max,
            value=8
        )

    with col2:
        storage_capacity = st.slider(
            "Storage Capacity (GB)",
            min_value=storage_min,
            max_value=storage_max,
            value=512
        )
        screen_size = st.slider(
            "Screen Size (inches)",
            min_value=round(screen_min, 1),
            max_value=round(screen_max, 1),
            value=round((screen_min + screen_max) / 2, 1)
        )
        weight = st.slider(
            "Weight (kg)",
            min_value=round(weight_min, 2),
            max_value=round(weight_max, 2),
            value=round((weight_min + weight_max) / 2, 2),
            step=0.1
        )

    if st.button(" Predict Price"):
        brand_encoded = le_brand.transform([brand_input])[0]

        input_data = pd.DataFrame([{
            "Brand": brand_encoded,
            "Processor_Speed": processor_speed,
            "RAM_Size": ram_size,
            "Storage_Capacity": storage_capacity,
            "Screen_Size": screen_size,
            "Weight": weight
        }])

        input_data = input_data[feature_cols]
        predicted_price = model.predict(input_data)[0]

        st.success(f" Estimated Laptop Price: **₹ {predicted_price:,.2f}**")

        st.image(
            "https://images.unsplash.com/photo-1517244861144-72a5b18b77c8",
            use_container_width=True,
            caption="Prediction generated based on your inputs "
        )

    st.markdown("---")
    st.subheader(" Sample Dataset")
    st.dataframe(df.head())


if __name__ == "__main__":
    main()
