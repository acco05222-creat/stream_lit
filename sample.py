import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
from faker import Faker
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import json
from streamlit_lottie import st_lottie

# --- 1. CONFIGURATION AND INITIAL SETUP ---
st.set_page_config(
    page_title="Real-Time Mortality Tracker Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Faker for generating realistic-looking data
fake = Faker()

# Setting up lottie json animation
path = "C:\\Users\\hp\\Desktop\\kaggle\\dtst\\animetion\\Employee Searching.json"
with open(path, 'r') as reed:
    motion = json.load(reed)
# Define constants for data generation
CAUSES_OF_DEATH = [
    'Natural Cause', 'Accident', 'Disease (Infectious)',
    'Disease (Chronic)', 'Suicide', 'Traffic Incident',
    'Workplace Accident', 'Fire', 'Other'
]

# Color mapping for causes
CAUSE_COLORS = {
    'Natural Cause': 'green',
    'Accident': 'red',
    'Disease (Infectious)': 'orange',
    'Disease (Chronic)': 'purple',
    'Suicide': 'darkred',
    'Traffic Incident': 'blue',
    'Workplace Accident': 'cadetblue',
    'Fire': 'darkblue',
    'Other': 'gray'
}


# --- 2. DATA GENERATION FUNCTION (Simulating Real-Time API) ---

@st.cache_data
def generate_synthetic_data(num_records=1000):
    """Generates a synthetic dataset for mortality records."""
    data = [ ]

    # Define a time window: last two months up to today
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=60)

    # Generate data
    for i in range(num_records):

        # Determine the date of death (more data recent days for "real-time" feel)
        death_date_dt = fake.date_time_between(start_date=start_date, end_date="now")

        cause = random.choice(CAUSES_OF_DEATH)

        # Make a small fraction of deaths today to highlight "real-time" effect
        if random.random() < 0.05:
            death_date_dt = fake.date_time_between(
                start_date=datetime.datetime.combine(today, datetime.time(0, 0)),
                end_date=datetime.datetime.now()
            )

        data.append({
            'ID': i + 1,
            'Name': fake.name(),
            'Age': random.randint(1, 95),
            'Cause': cause,
            'Date': death_date_dt.date(),
            'Time': death_date_dt.time().strftime('%H:%M:%S'),
            'Timestamp': death_date_dt,
            'Latitude': fake.latitude(),
            'Longitude': fake.longitude(),
            'Location': fake.city() + ", " + fake.country(),
            # Mock source simulation
            'Source': random.choice(
                ['News Report', 'Social Media Alert', 'Public Record', 'Police Log', 'Family Report'])
        })

    df = pd.DataFrame(data)
    # Convert lat/lon to numeric for map plotting
    df['Latitude'] = pd.to_numeric(df['Latitude'])
    df['Longitude'] = pd.to_numeric(df['Longitude'])

    return df


# Load data (simulated real-time records)
df_records = generate_synthetic_data(num_records=2500)


# --- 3. Sidebar ---

st.sidebar.header("Dashboard :red[Filters]", divider='gray')

# 3.1. Date Filter (st.date_input)
today = datetime.date.today()
two_months_ago = today - datetime.timedelta(days=60)

st.sidebar.subheader("Date Range")
date_selection = st.sidebar.date_input(
    "Select Date Range",
    value=(two_months_ago, today),
    min_value=df_records['Date'].min(),
    max_value=today,
    help="Select the start and end dates to filter mortality records"
)

if len(date_selection) == 2:
    start_date = min(date_selection)
    end_date = max(date_selection)

    # Filter by date range
    df_filtered_date = df_records[
        (df_records['Date'] >= start_date) & (df_records['Date'] <= end_date)
        ].copy()
else:
    # If only one date is selected (or being selected), show all data in the range
    df_filtered_date = df_records.copy()

st.sidebar.markdown("---")

# 3.2. Cause of Death Filter - SINGLE SELECT
st.sidebar.subheader("Cause of Death")

# Add an 'All Causes' option to the beginning of the list for easy selection
ALL_CAUSES_OPTION = "All Causes"
cause_options = [ALL_CAUSES_OPTION] + CAUSES_OF_DEATH

# Use st.selectbox for single selection
selected_cause = st.sidebar.selectbox(
    "Choose one cause:",
    options=cause_options,
    index=0,  # Default to 'All Causes'
    help="Filter records by a single cause of death."
)

# Filtering logic for single select
if selected_cause == ALL_CAUSES_OPTION:
    # If 'All Causes' is selected, include all records from the date filter
    df_filtered = df_filtered_date.copy()

    # For the legend, treat all causes as "selected"
    selected_causes_for_legend = CAUSES_OF_DEATH
else:
    # Filter by the single selected cause
    df_filtered = df_filtered_date[
        df_filtered_date['Cause'] == selected_cause
    ].copy()
    # For the legend, only the selected cause is "selected"
    selected_causes_for_legend = [selected_cause]


# Display color legend for causes (using the list of "active" causes)
st.sidebar.markdown("---")
st.sidebar.subheader("Color code")
legend_html = "<div style='font-size: 12px;'>"
# Only show the color for the active cause(s)
for cause, color in CAUSE_COLORS.items():
    if cause in selected_causes_for_legend:
        legend_html += f"<p style='margin: 2px 0;'><span style='color:{color}; font-size: 16px;'>⬤</span> {cause}</p>"
legend_html += "</div>"
st.sidebar.markdown(legend_html, unsafe_allow_html=True)

selected_causes = selected_causes_for_legend

st.sidebar.markdown("---")
st.sidebar.success(f"Showing **{len(df_filtered):,}** records")
st.sidebar.info(f"Date Range: {start_date} to {end_date}")

# ... (the rest of the app code is unchanged)

# --- 4. MAIN LAYOUT AND METRICS ---
col_1, col_2 = st.columns(2)
with col_1:
    st_lottie(motion, speed=1, loop=True, width=300)
with col_2:
    st.title("Simulated Global Mortality Record ")
st.caption("Displaying synthetic data generated to simulate real-time death records from various sources.")

col1, col2, col3 = st.columns(3)

# 4.1. Metric: Total Death (Today)
df_today = df_records[df_records['Date'] == today]
total_today = len(df_today)
# Calculate the total from yesterday for delta
yesterday = today - datetime.timedelta(days=1)
total_yesterday = len(df_records[df_records['Date'] == yesterday])
delta_today = total_today - total_yesterday

col1.metric(
    "Total Deaths (Today)",
    f"{total_today:,}",
    delta=f"{delta_today:+d} vs yesterday",
    delta_color="normal" if delta_today <= 0 else "inverse"
, border = True)

# 4.2. Metric: Total Death (Last 30 Days)
last_30_days = today - datetime.timedelta(days=30)
df_month = df_records[df_records['Date'] >= last_30_days]
total_month = len(df_month)

# Calculate total from the 30 days prior to the last 30 days for delta
prior_30_days_start = last_30_days - datetime.timedelta(days=30)
df_prior_month = df_records[
    (df_records['Date'] >= prior_30_days_start) & (df_records['Date'] < last_30_days)
    ]
total_prior_month = len(df_prior_month)
delta_month = total_month - total_prior_month

col2.metric(
    "Total Deaths (Last 30 Days)",
    f"{total_month:,}",
    delta=f"{delta_month:+d} vs prior month",
    delta_color="normal" if delta_month <= 0 else "inverse"
, border = True)

# 4.3. Mortality Trending (Sparkline Chart)
df_trend = df_records.groupby(df_records['Date']).size().reset_index(name='Count')
df_trend['Date'] = pd.to_datetime(df_trend['Date'])
df_trend = df_trend.set_index('Date').sort_index()
# Use a series of the last 60 days of counts for the sparkline
chart_data = df_trend['Count'].tail(60).tolist()

col3.metric(
    "7-Day Trend (Avg Daily)",
    f"{df_trend['Count'].tail(7).mean():.1f}",
    delta=None,
    help="Average number of deaths per day over the last 7 days."
, border = True, chart_data= df_trend, chart_type= 'area')

st.markdown("---")

# --- 5. VISUALIZATION AND TRENDING ---

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs([" Map View", "Trends & Analytics", "Event Log"])

# 5.1. --------------------- Map Visualization  -------------------
with tab1:
    st.subheader("Geographic Distribution of Mortality Records")

    if not df_filtered.empty and len(selected_causes) > 0:
        # Calculate central point for map view
        avg_lat = df_filtered['Latitude'].mean()
        avg_lon = df_filtered['Longitude'].mean()

        # Create the folium map
        m = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=2,
            tiles='OpenStreetMap'
        )

        # Create a MarkerCluster for better performance with many points
        marker_cluster = MarkerCluster(
            name="Mortality Records",
            overlay=True,
            control=True,
        ).add_to(m)

        # Add markers for each record
        for index, row in df_filtered.iterrows():
            # Get color based on cause
            marker_color = CAUSE_COLORS.get(row['Cause'], 'gray')

            popup_html = f"""
            <div style="width: 250px; font-family: Arial;">
                <h4 style="margin: 0 0 10px 0; color: {marker_color};">{row['Cause']}</h4>
                <p style="margin: 3px 0;"><b>Name:</b> {row['Name']}</p>
                <p style="margin: 3px 0;"><b>Age:</b> {row['Age']} years</p>
                <p style="margin: 3px 0;"><b>Date:</b> {row['Date']}</p>
                <p style="margin: 3px 0;"><b>Time:</b> {row['Time']}</p>
                <p style="margin: 3px 0;"><b>Location:</b> {row['Location']}</p>
                <p style="margin: 3px 0;"><b>Source:</b> {row['Source']}</p>
            </div>
            """

            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=6,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{row['Cause']} - {row['Name']}",
                color=marker_color,
                fill=True,
                fillColor=marker_color,
                fillOpacity=0.7,
                weight=2
            ).add_to(marker_cluster)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Display the map using st_folium
        st_folium(m, width=None, height=600, returned_objects=[])

        st.info(f"Displaying {len(df_filtered):,} mortality records on the map")
    else:
        st.warning("⚠️ No records match the current filters to display on the map. Please adjust your filters.")

# 5.2. -------------  Mortality Trending Chart:TAB 2 --------------
with tab2:
    col_chart1, col_chart2 = st.columns(2, vertical_alignment='bottom')

    with col_chart1:
        st.subheader("Daily Mortality Trend")

        # Recalculate daily counts based on the date filter
        df_daily_counts = df_filtered_date.groupby('Date').size().reset_index(name='Daily Count')
        df_daily_counts['Date'] = pd.to_datetime(df_daily_counts['Date'])

        # Use st.line_chart for a built-in interactive chart
        if not df_daily_counts.empty:
            st.line_chart(df_daily_counts.set_index('Date'), height=400)
        else:
            st.info("🪧 Select a date range with data to see the trend.")

    with col_chart2:
        st.subheader("Deaths by Cause")

        if not df_filtered.empty:
            cause_counts = df_filtered['Cause'].value_counts()
            st.bar_chart(cause_counts, height=400)
        else:
            st.info("🪧 No data to display.")

    # Additional analytics
    st.markdown("---")
    st.subheader("Age Distribution")

    if not df_filtered.empty:
        # Create age groups
        df_filtered['Age_Group'] = pd.cut(
            df_filtered['Age'],
            bins=[0, 18, 35, 50, 65, 100],
            labels=['0-18', '19-35', '36-50', '51-65', '65+']
        )

        age_group_counts = df_filtered['Age_Group'].value_counts().sort_index()

        col_age1, col_age2, col_age3 = st.columns(3)

        with col_age1:
            st.bar_chart(age_group_counts, height=300)

        with col_age2:
            st.subheader("Top 5 Locations")
            top_locations = df_filtered['Location'].value_counts().head(5)
            st.dataframe(top_locations, use_container_width=True)

        with col_age3:
            st.subheader("Sources")
            source_counts = df_filtered['Source'].value_counts()
            st.dataframe(source_counts, use_container_width=True)

# 5.3. --------------- Event Log Table :TAB 3 -----------
with tab3:
    st.subheader("Event Log: Detailed Records")

    # Add search functionality
    search_term = st.text_input("Search records (by name, location, cause)", "", placeholder='Enter text')

    # Select and rename columns for display
    df_display = df_filtered[[
        'ID', 'Name', 'Age', 'Cause', 'Date', 'Time', 'Location', 'Source', 'Latitude', 'Longitude'
    ]].rename(columns={
        'Cause': 'Cause of Death',
        'Date': 'Date (Y-M-D)',
        'Time': 'Time (H:M:S)',
        'Location': 'Approximate Location'
    })

    # Apply search filter if search term exists
    if search_term:
        mask = df_display.astype(str).apply(
            lambda row: row.str.contains(search_term, case=False).any(),
            axis=1
        )
        df_display = df_display[mask]

    # Display record count
    st.info(f"Showing {len(df_display):,} records")

    # Display dataframe with improved styling
    st.dataframe(
        df_display,
        use_container_width=True,
        height=500,
        hide_index=False
    )

    # Download button
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Event Log (CSV)",
        data=csv,
        file_name=f"mortality_records_{datetime.date.today()}.csv",
        mime="text/csv",
        use_container_width=True
    )

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><b>Global Mortality Records Dashboard</b></p>
    <p>⚠️ This dashboard uses synthetic data for demonstration purposes only</p>
    <p>Data sources: Simulated news reports, social media alerts, public records, and official logs</p>
    <p style='font-size: 12px;'>Last updated: {}</p>
</div>
""".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

