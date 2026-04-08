import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Automobile Analysis", layout="wide")

st.title("Automobile Data Analysis Dashboard")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    headers = ["symboling", "normalized-losses", "make", 
               "fuel-type", "aspiration","num-of-doors",
               "body-style","drive-wheels", "engine-location",
               "wheel-base","length", "width","height", "curb-weight",
               "engine-type","num-of-cylinders", "engine-size", 
               "fuel-system","bore","stroke", "compression-ratio",
               "horsepower", "peak-rpm","city-mpg","highway-mpg","price"]

    df = df.iloc[:, :len(headers)]
    df.columns = headers

    df.replace("?", np.nan, inplace=True)

    for col in ["price", "horsepower", "peak-rpm", "city-mpg", "engine-size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["price", "horsepower", "engine-size", "city-mpg"], inplace=True)

    df["length"] = df["length"].astype(float) / df["length"].astype(float).max()
    df["width"] = df["width"].astype(float) / df["width"].astype(float).max()
    df["height"] = df["height"].astype(float) / df["height"].astype(float).max()

    df["city-L/100km"] = 235 / df["city-mpg"]

    bins = np.linspace(df["price"].min(), df["price"].max(), 4)
    df["price-binned"] = pd.cut(df["price"], bins=bins,
                               labels=['Low', 'Medium', 'High'],
                               include_lowest=True)

    return df

data = load_data()

# ================= FILTERS =================
st.sidebar.header("Filters")

make = st.sidebar.multiselect("Make", data["make"].dropna().unique(), default=data["make"].dropna().unique())
fuel = st.sidebar.multiselect("Fuel Type", data["fuel-type"].dropna().unique(), default=data["fuel-type"].dropna().unique())
body = st.sidebar.multiselect("Body Style", data["body-style"].dropna().unique(), default=data["body-style"].dropna().unique())
drive = st.sidebar.multiselect("Drive Wheels", data["drive-wheels"].dropna().unique(), default=data["drive-wheels"].dropna().unique())
engine = st.sidebar.multiselect("Engine Type", data["engine-type"].dropna().unique(), default=data["engine-type"].dropna().unique())
cylinders = st.sidebar.multiselect("Cylinders", data["num-of-cylinders"].dropna().unique(), default=data["num-of-cylinders"].dropna().unique())

price_range = st.sidebar.slider("Price", int(data["price"].min()), int(data["price"].max()),
                               (int(data["price"].min()), int(data["price"].max())))

engine_range = st.sidebar.slider("Engine Size", int(data["engine-size"].min()), int(data["engine-size"].max()),
                                (int(data["engine-size"].min()), int(data["engine-size"].max())))

hp_range = st.sidebar.slider("Horsepower", int(data["horsepower"].min()), int(data["horsepower"].max()),
                            (int(data["horsepower"].min()), int(data["horsepower"].max())))

mpg_range = st.sidebar.slider("City MPG", int(data["city-mpg"].min()), int(data["city-mpg"].max()),
                             (int(data["city-mpg"].min()), int(data["city-mpg"].max())))

filtered_data = data[
    (data["make"].isin(make)) &
    (data["fuel-type"].isin(fuel)) &
    (data["body-style"].isin(body)) &
    (data["drive-wheels"].isin(drive)) &
    (data["engine-type"].isin(engine)) &
    (data["num-of-cylinders"].isin(cylinders)) &
    (data["price"].between(price_range[0], price_range[1])) &
    (data["engine-size"].between(engine_range[0], engine_range[1])) &
    (data["horsepower"].between(hp_range[0], hp_range[1])) &
    (data["city-mpg"].between(mpg_range[0], mpg_range[1]))
]

# ================= DATA =================
st.subheader("Dataset Preview")
st.dataframe(filtered_data.head())

# ================= KPI =================
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Average Price", int(filtered_data["price"].mean()) if not filtered_data.empty else 0)
col2.metric("Max Price", int(filtered_data["price"].max()) if not filtered_data.empty else 0)
col3.metric("Total Cars", filtered_data.shape[0])

# ================= VISUALIZATION =================
if not filtered_data.empty:

    fig1, ax1 = plt.subplots()
    sns.boxplot(x='drive-wheels', y='price', data=filtered_data, ax=ax1)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='engine-size', y='price', hue='drive-wheels', data=filtered_data, ax=ax2)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.histplot(filtered_data["price"], kde=True, ax=ax3)
    st.pyplot(fig3)

    avg_price = filtered_data.groupby("make")["price"].mean().sort_values()
    fig4, ax4 = plt.subplots()
    avg_price.plot(kind='bar', ax=ax4)
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots(figsize=(10,6))
    sns.heatmap(filtered_data.corr(numeric_only=True), annot=True, ax=ax5)
    st.pyplot(fig5)

    fig6, ax6 = plt.subplots()
    sns.regplot(x='engine-size', y='price', data=filtered_data, ax=ax6)
    st.pyplot(fig6)

# ================= STATISTICS =================
st.subheader("Statistical Comparison")

selected_brands = st.multiselect("Select brands", options=filtered_data["make"].unique())

subset = filtered_data[filtered_data["make"].isin(selected_brands)]

if len(selected_brands) >= 2 and not subset.empty:

    fig7, ax7 = plt.subplots()
    sns.boxplot(x='make', y='price', data=subset, ax=ax7)
    st.pyplot(fig7)

    groups = [group['price'].values for name, group in subset.groupby('make') if len(group) > 1]

    if len(groups) == 2:
        result = stats.ttest_ind(groups[0], groups[1])
        st.write("T-test:", result)

    elif len(groups) >= 3:
        result = stats.f_oneway(*groups)
        st.write("ANOVA:", result)

# ================= ML MODEL =================
st.subheader("Machine Learning Model")

def prepare_ml(df):
    df_ml = df.copy()
    le_dict = {}

    cat_cols = ["make","fuel-type","aspiration","body-style",
                "drive-wheels","engine-type","num-of-cylinders"]

    for col in cat_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        le_dict[col] = le

    features = cat_cols + ["engine-size","horsepower","city-mpg"]
    X = df_ml[features]
    y = df_ml["price"]

    return X, y, le_dict, features

X, y, le_dict, features = prepare_ml(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write("R2 Score:", r2_score(y_test, y_pred))
st.write("MAE:", mean_absolute_error(y_test, y_pred))

# Prediction UI
st.subheader("Predict Price")

input_data = {}

for col in ["make","fuel-type","aspiration","body-style",
            "drive-wheels","engine-type","num-of-cylinders"]:
    val = st.selectbox(col, le_dict[col].classes_)
    input_data[col] = le_dict[col].transform([val])[0]

input_data["engine-size"] = st.slider("Engine Size", 50, 300, 120)
input_data["horsepower"] = st.slider("Horsepower", 50, 300, 100)
input_data["city-mpg"] = st.slider("City MPG", 10, 50, 25)

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Price: {int(pred)}")

# Feature importance
importance = pd.Series(model.coef_, index=features).sort_values()
fig8, ax8 = plt.subplots()
importance.plot(kind='barh', ax=ax8)
st.pyplot(fig8)

# ================= DOWNLOAD =================
st.download_button("Download Data", filtered_data.to_csv(index=False), "data.csv")