import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

st.set_page_config(page_title="Automobile Analysis", layout="wide")

st.title("Automobile Data Analysis Dashboard")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")

    # Remove unwanted index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    headers = ["symboling", "normalized-losses", "make", 
               "fuel-type", "aspiration","num-of-doors",
               "body-style","drive-wheels", "engine-location",
               "wheel-base","length", "width","height", "curb-weight",
               "engine-type","num-of-cylinders", "engine-size", 
               "fuel-system","bore","stroke", "compression-ratio",
               "horsepower", "peak-rpm","city-mpg","highway-mpg","price"]

    # Fix column mismatch safely
    df = df.iloc[:, :len(headers)]
    df.columns = headers

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Drop missing price
    df.dropna(subset=["price"], inplace=True)

    # Convert numeric safely
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df["peak-rpm"] = pd.to_numeric(df["peak-rpm"], errors="coerce")
    df["city-mpg"] = pd.to_numeric(df["city-mpg"], errors="coerce")

    df.dropna(subset=["price"], inplace=True)

    # Normalize
    df["length"] = df["length"].astype(float) / df["length"].astype(float).max()
    df["width"] = df["width"].astype(float) / df["width"].astype(float).max()
    df["height"] = df["height"].astype(float) / df["height"].astype(float).max()

    # Convert mpg → L/100km
    df["city-L/100km"] = 235 / df["city-mpg"]

    # Binning
    bins = np.linspace(df["price"].min(), df["price"].max(), 4)
    group_names = ['Low', 'Medium', 'High']
    df["price-binned"] = pd.cut(df["price"], bins=bins, labels=group_names, include_lowest=True)

    return df

# Load data
data = load_data()

# Sidebar filters
st.sidebar.header("Filters")
drive = st.sidebar.multiselect(
    "Drive Wheels",
    options=data["drive-wheels"].dropna().unique(),
    default=data["drive-wheels"].dropna().unique()
)

filtered_data = data[data["drive-wheels"].isin(drive)]

# Show data
st.subheader("Dataset Preview")
st.dataframe(filtered_data.head())

# Boxplot
st.subheader("Price vs Drive Wheels")
fig1, ax1 = plt.subplots()
sns.boxplot(x='drive-wheels', y='price', data=filtered_data, ax=ax1)
st.pyplot(fig1)

# Scatter plot
st.subheader("Engine Size vs Price")
fig2, ax2 = plt.subplots()
ax2.scatter(filtered_data['engine-size'], filtered_data['price'])
ax2.set_xlabel("Engine Size")
ax2.set_ylabel("Price")
ax2.grid()
st.pyplot(fig2)

# Histogram
st.subheader("Price Distribution")
fig3, ax3 = plt.subplots()
ax3.hist(filtered_data["price"], bins=20)
st.pyplot(fig3)

# Grouped data
st.subheader("Grouped Mean Price")
grouped = filtered_data.groupby(['drive-wheels', 'body-style'], as_index=False)['price'].mean()
st.dataframe(grouped)

# ANOVA
st.subheader("ANOVA Test (Honda vs Subaru)")
try:
    grouped_annova = data.groupby('make')
    anova = sp.stats.f_oneway(
        grouped_annova.get_group('honda')['price'],
        grouped_annova.get_group('subaru')['price']
    )
    st.write("F-value:", anova.statistic)
    st.write("P-value:", anova.pvalue)
except:
    st.warning("Not enough data for ANOVA comparison.")

# Regression plot
st.subheader("Regression Plot")
fig4, ax4 = plt.subplots()
sns.regplot(x='engine-size', y='price', data=filtered_data, ax=ax4)
st.pyplot(fig4)

st.success("App is running successfully.")