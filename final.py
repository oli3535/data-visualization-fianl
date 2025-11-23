import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import pydeck as pdk

# Page configuration
st.set_page_config(
    page_title="Los Angeles Crime Data Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading with optimized parameters
@st.cache_data(ttl=3600, show_spinner=True)
def load_data():
    usecols = ['DATE OCC', 'TIME OCC', 'AREA NAME', 'Crm Cd Desc', 'Vict Age', 
              'Vict Sex', 'Status Desc', 'LAT', 'LON', 'Weapon Desc']
    data = pd.read_csv('Crime_Data_from_2020_to_Present.parquet.csv', 
                      usecols=usecols, 
                      low_memory=False)
    return data

# Sidebar
with st.sidebar:
    st.write("**Created by Zhang Hanzhong**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("WUT-Logo.png", caption="Wuhan University of Technology")
    with col2:
        st.image("efrei.png", caption="EFREI Paris")
    
    st.markdown("---")
    st.write("### Data Source")
    st.write("Los Angeles Crime Data from 2020 to Present")

# Main content
st.title(" Los Angeles Crime Data Analysis (2020-Present)")
st.markdown("---")

# Part 1: Introduction
st.header(" Introduction")
st.write("""
This dashboard provides an interactive analysis of crime data in Los Angeles from 2020 to present. 
The analysis aims to identify crime patterns, victims, and geographical distributions to 
support public safety initiatives and resource allocation.
""")

# Load data
st.header(" Data Loading & Cleaning")
df = load_data()
st.success(f"Data loaded successfully! Dataset shape: {df.shape}")

# Display raw data
with st.expander("View Raw Data Sample"):
    st.dataframe(df.head(10))

# Part 2: Data Sources
st.header(" Data Sources")
st.write("""
**Primary Dataset:** Crime Data from 2020 to Present
- **Source:** Los Angeles Police Department
- **Format:** CSV
- **Time Period:** 2020 - Present
- **Records:** Approximately 1 million+ incidents

The dataset includes comprehensive information about each crime incident including:
- Crime type and description
- Date and time of occurrence
- Geographical location (latitude/longitude)
- Victim demographics
- Police area information
- Case status
""")


# Data preprocessing
st.subheader("Data Preprocessing")

# Clean the data
df_clean = df.copy()

# Handle date and time columns
df_clean['DATE OCC'] = pd.to_datetime(df_clean['DATE OCC'], errors='coerce')
df_clean = df_clean.dropna(subset=['DATE OCC'])

# Extract temporal features
df_clean['year'] = df_clean['DATE OCC'].dt.year
df_clean['month'] = df_clean['DATE OCC'].dt.month
df_clean['day_of_week'] = df_clean['DATE OCC'].dt.day_name()

# Handle time column
df_clean['TIME OCC'] = pd.to_numeric(df_clean['TIME OCC'], errors='coerce')
df_clean['hour'] = (df_clean['TIME OCC'] // 100).astype('Int64')

# Clean geographical data
df_clean = df_clean.dropna(subset=['LAT', 'LON'])
df_clean = df_clean[(df_clean['LAT'] != 0) & (df_clean['LON'] != 0)]
df_clean = df_clean[(df_clean['LAT'].between(33, 35)) & (df_clean['LON'].between(-119, -117))]

# Remove duplicate records
initial_count = len(df_clean)
df_clean = df_clean.drop_duplicates()
duplicates_removed = initial_count - len(df_clean)

# Handle missing values in critical columns
critical_columns = ['AREA NAME', 'Crm Cd Desc', 'Vict Sex', 'Status Desc']
initial_count_2 = len(df_clean)
df_clean = df_clean.dropna(subset=critical_columns)
missing_removed = initial_count_2 - len(df_clean)

st.success(f"Data cleaned! Remaining records: {len(df_clean)}")
# Data cleaning steps
st.write("**Data Cleaning Steps Applied:**")
cleaning_steps = [
    " Handled missing values in critical columns",
    " Removed duplicate records"    
]

for step in cleaning_steps:
    st.write(step)

# Part 3: Exploratory Data Analysis
st.header("  Data Analysis")
st.markdown("---")



# Question 1: Crime trends over time - Using existing year and month fields
# Question 1: Top crime areas
st.write("**1. Which areas have the highest crime rates?**")

# Top 10 areas with most crimes
area_crimes = df_clean['AREA NAME'].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 6))
area_crimes.plot(kind='barh', ax=ax, color='coral')
ax.set_title('Top 10 Areas with Highest Crime Rates')
ax.set_xlabel('Number of Crimes')
plt.tight_layout()
st.pyplot(fig)

# Question 2: Crime types distribution
st.write("**2. What are the most common types of crimes?**")

top_crimes = df_clean['Crm Cd Desc'].value_counts().head(10)

fig2, ax2 = plt.subplots(figsize=(10, 6))
top_crimes.plot(kind='barh', ax=ax2)
ax2.set_title('Top 10 Most Common Crime Types')
ax2.set_xlabel('Number of Incidents')
plt.tight_layout()
st.pyplot(fig2)

# Question 3: Crime Status Analysis
st.write("**3. What are the status outcomes of reported crimes?**")

status_counts = df_clean['Status Desc'].value_counts()

fig3, ax3 = plt.subplots(figsize=(10, 6))
status_counts.plot(kind='bar', ax=ax3, color='lightgreen')
ax3.set_title('Crime Case Status Distribution')
ax3.set_xlabel('Case Status')
ax3.set_ylabel('Number of Cases')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig3)

# Question 4: Weapon Usage in Crimes
st.write("**4. What weapons are most commonly used in crimes?**")

weapon_counts = df_clean['Weapon Desc'].value_counts().head(10)

fig4, ax4 = plt.subplots(figsize=(10, 6))
weapon_counts.plot(kind='barh', ax=ax4, color='purple')
ax4.set_title('Top 10 Weapons Used in Crimes')
ax4.set_xlabel('Number of Incidents')
plt.tight_layout()
st.pyplot(fig4)


# Sample data for maps to avoid size limits
map_sample = df_clean.sample(n=min(5000, len(df_clean)), random_state=42)

# Visualization 5: Crime Heat Map
st.write("**5. Where are the crime hotspots located in Los Angeles?**")

heatmap_layer = pdk.Layer(
    'HeatmapLayer',
    data=map_sample,
    get_position=['LON', 'LAT'],
    radius=100,
    intensity=1,
    threshold=0.3,
)

view_state = pdk.ViewState(
    latitude=34.0522,
    longitude=-118.2437,
    zoom=10,
    pitch=0,
)

st.pydeck_chart(pdk.Deck(
    layers=[heatmap_layer],
    initial_view_state=view_state,
    map_style='mapbox://styles/mapbox/light-v10'
))


# Visualization 6: Crime Types Distribution Across Areas
st.write("**6. How are different crime types distributed across police areas?**")

# Get top 5 crime types overall
top_5_crimes = df_clean['Crm Cd Desc'].value_counts().head(5).index

# Create a pivot table of crime counts by area and crime type
crime_by_area = pd.crosstab(df_clean['AREA NAME'], df_clean['Crm Cd Desc'])
top_crimes_by_area = crime_by_area[top_5_crimes]

# Create stacked bar chart
fig6, ax6 = plt.subplots(figsize=(12, 8))
top_crimes_by_area.plot(kind='bar', ax=ax6, stacked=True)
ax6.set_title('Distribution of Top 5 Crime Types Across Police Areas')
ax6.set_xlabel('Police Area')
ax6.set_ylabel('Number of Crimes')
ax6.legend(title='Crime Types', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig6)

        

# 7
st.write("**7. What are the victim patterns?**")




st.write("**Victim Age Distribution**")
age_data = df_clean['Vict Age'].dropna()
age_data = age_data[(age_data > 0) & (age_data < 100)]
    
fig5, ax5 = plt.subplots(figsize=(8, 4))
ax5.hist(age_data, bins=30, alpha=0.7, color='skyblue')
ax5.set_xlabel('Victim Age')
ax5.set_ylabel('Frequency')
plt.tight_layout()
st.pyplot(fig5)


# Victim demographics
st.write("**Victim Sex Distribution**")
victim_sex = df_clean['Vict Sex'].value_counts()
fig7, ax7 = plt.subplots(figsize=(8, 4))
victim_sex.plot(kind='bar', ax=ax7, color=['lightblue', 'lightpink', 'lightgray'])
ax7.set_title('Victim Gender Distribution')
ax7.set_xlabel('Gender')
ax7.set_ylabel('Count')
plt.tight_layout()
st.pyplot(fig7)

