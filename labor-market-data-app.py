import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import seaborn as sns
import folium
import streamlit_folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Analysis of UK Labor Market Data",layout="wide",initial_sidebar_state="expanded")

def load_and_preprocess_data():
    df = pd.read_csv("dfall_clean.csv")
    
    df = df[(df['salary_mid'] > 1000) | (df['salary_mid'].isna())]
    
    df['created_dt'] = pd.to_datetime(df['created_dt'], errors='coerce')
    df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
    df['month'] = df['created_dt'].dt.to_period('M')
    
    df['region'] = df['location_area'].str.extract(r'UK > ([^ >]+)')
    df['region'] = df['region'].fillna('Unknown')
    
    df['contract_type'] = df['contract_type'].fillna('Unknown')
    df['contract_time'] = df['contract_time'].fillna('Unknown')
    
    category_counts = df['category_label'].value_counts()
    main_categories = category_counts[category_counts > 20].index.tolist()
    df['main_category'] = df['category_label'].where(df['category_label'].isin(main_categories), 'Other')
    
    return df

df = load_and_preprocess_data()

st.sidebar.header("Data filtering conditions")

salary_min, salary_max = st.sidebar.slider(
    "Salary range (annual, GBP)",
    min_value=int(df['salary_mid'].min()),
    max_value=int(df['salary_mid'].max()),
    value=(20000, 80000),
    step=5000
)

selected_categories = st.sidebar.multiselect(
    "Job Categories",
    options=df['main_category'].unique(),
    default=['IT Jobs', 'Teaching Jobs', 'Healthcare & Nursing Jobs', 'Logistics & Warehouse Jobs']
)

contract_time = st.sidebar.radio(
    "Contract Type",
    options=['All', 'full_time', 'part_time'],
    index=0
)

filtered_df = df[
    (df['salary_mid'] >= salary_min) & 
    (df['salary_mid'] <= salary_max) & 
    (df['main_category'].isin(selected_categories))
]

if contract_time != 'All':
    filtered_df = filtered_df[filtered_df['contract_time'] == contract_time]

st.title("UK Labour Market Data Analysis Report")
st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total number of positions", f"{len(filtered_df):,}")
with col2:
    st.metric("Average salary", f"£{filtered_df['salary_mid'].mean():.0f}")
with col3:
    st.metric("Number of regions", filtered_df['region'].nunique())
with col4:
    st.metric("Data time range", f"{filtered_df['created_date'].min().strftime('%Y-%m')}~{filtered_df['created_date'].max().strftime('%Y-%m')}")

st.divider()

st.header("Comparison of salary levels for different job categories")

st.subheader("Box plot of salary distribution by job category")
fig1, ax1 = plt.subplots(figsize=(12, 6))
filtered_df_box = filtered_df[filtered_df['main_category'].isin(selected_categories)]

category_avg_salary = filtered_df_box.groupby('main_category')['salary_mid'].mean().sort_values(ascending=False)
ordered_categories = category_avg_salary.index.tolist()

box_plot = ax1.boxplot(
    [filtered_df_box[filtered_df_box['main_category'] == cat]['salary_mid'] for cat in ordered_categories],
    labels=ordered_categories,
    patch_artist=True,
    showfliers=False 
)

colors = plt.cm.Set3(np.linspace(0, 1, len(ordered_categories)))
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

ax1.set_xlabel("Job Category", fontsize=12)
ax1.set_ylabel("Salary (GBP/year)", fontsize=12)
ax1.set_title("Salary Distribution by Job Category (Median, Quartiles)", fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig1)

st.subheader("Bar chart of average salary by job category")
fig2, ax2 = plt.subplots(figsize=(12, 6))

avg_salary_by_category = filtered_df.groupby('main_category')['salary_mid'].agg(['mean', 'count']).sort_values('mean', ascending=False)
avg_salary_by_category = avg_salary_by_category[avg_salary_by_category['count'] >= 5]  # 过滤样本数<5的类别

bars = ax2.bar(
    avg_salary_by_category.index,
    avg_salary_by_category['mean'],
    color=plt.cm.viridis(np.linspace(0, 1, len(avg_salary_by_category))),
    alpha=0.8
)

for i, (idx, row) in enumerate(avg_salary_by_category.iterrows()):
    ax2.text(
        i, row['mean'] + 1000,
        f"£{row['mean']:.0f}\n(n={row['count']})",
        ha='center', va='bottom', fontsize=10
    )

ax2.set_xlabel("Job Category", fontsize=12)
ax2.set_ylabel("Average Salary (GBP/year)", fontsize=12)
ax2.set_title("Average Salary Comparison by Job Category (Sample Size ≥ 5)", fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig2)

st.write("### Conclusion 1: Salary Characteristics of Job Categories")
st.markdown(f"""
- **Top 3 High-Paying Positions**: {avg_salary_by_category.head(3).index.tolist()} with average salaries exceeding £{avg_salary_by_category.head(3)['mean'].min():.0f}
- **Salary Disparity**: The salary gap between the highest and lowest paying categories is approximately £{avg_salary_by_category['mean'].max() - avg_salary_by_category['mean'].min():.0f}
- **Stability**: The salary distribution for {ordered_categories[0]} is the most concentrated (narrowest in the box plot), while {ordered_categories[-1]} has the greatest salary fluctuation
""")

st.divider()

st.header("Geographical Distribution of Job Positions and Regional Salary Levels")

st.subheader("Geographical Distribution Map of Job Positions")
map_df = filtered_df.dropna(subset=['latitude', 'longitude'])

if len(map_df) > 0:
    uk_center = [55.3781, -3.4360]
    m = folium.Map(location=uk_center, zoom_start=6)
    
    category_color_map = {cat: plt.cm.tab10(i) for i, cat in enumerate(selected_categories)}
    
    for idx, row in map_df.iterrows():
        popup_content = f"""
        <b>Job Title</b>: {row['title']}<br>
        <b>Company</b>: {row['company_name']}<br>
        <b>Category</b>: {row['main_category']}<br>
        <b>Salary</b>: £{row['salary_mid']:.0f}/year<br>
        <b>Location</b>: {row['location_display']}
        """
        
        salary_normalized = (row['salary_mid'] - salary_min) / (salary_max - salary_min)
        radius = 5 + salary_normalized * 10  # 半径范围5-15
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=category_color_map.get(row['main_category'], '#888888'),
            fill=True,
            fill_color=category_color_map.get(row['main_category'], '#888888'),
            fill_opacity=0.6,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(m)

    st_folium(m, width=1200, height=600)
else:
    st.warning("There is no valid latitude and longitude data under the current filtering conditions, and the map cannot be displayed")

st.subheader("Histogram of salary distribution in major regions")
fig3, ax3 = plt.subplots(figsize=(12, 6))

region_counts = filtered_df['region'].value_counts()
main_regions = region_counts[region_counts > 10].index.tolist()
region_salary_df = filtered_df[filtered_df['region'].isin(main_regions)]

for i, region in enumerate(main_regions):
    region_data = region_salary_df[region_salary_df['region'] == region]['salary_mid']
    if len(region_data) > 0:
        ax3.hist(
            region_data,
            alpha=0.6,
            bins=15,
            label=f"{region} (n={len(region_data)})",
            color=plt.cm.Set2(i % 8)
        )

ax3.set_xlabel("Salary (GBP/year)", fontsize=12)
ax3.set_ylabel("Number of positions", fontsize=12)
ax3.set_title("Salary Distribution Frequency by Region", fontsize=14, fontweight='bold')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(axis='y', alpha=0.3)
st.pyplot(fig3)

st.subheader("Salary Statistics in Major Regions")
region_salary_stats = filtered_df.groupby('region')['salary_mid'].agg([
    'count', 'mean', 'median', 'std'
]).round(0)
region_salary_stats = region_salary_stats[region_salary_stats['count'] >= 5].sort_values('mean', ascending=False)
region_salary_stats.columns = ['Number of Positions', 'Average Salary', 'Median Salary', 'Salary Std Dev']
st.dataframe(region_salary_stats)

st.write("### Conclusion 2: Geographic distribution and salary characteristics")
if len(region_salary_stats) > 0:
    st.markdown(f"""
    - **Top 3 High-Paying Regions**: {region_salary_stats.head(3).index.tolist()} with average salaries exceeding £{region_salary_stats.head(3)['Average Salary'].min():.0f}
    - **Job Density**: {region_counts.index[0]} has the most positions ({region_counts.iloc[0]}), accounting for {region_counts.iloc[0]/len(filtered_df)*100:.1f}% of total positions
    - **Salary Stability**: {region_salary_stats['Salary Std Dev'].idxmin()} has the most stable salaries (standard deviation £{region_salary_stats['Salary Std Dev'].min():.0f})
    """)

st.divider()

st.header("Analysis of Time Trends in the Labor Market")

trend_df = filtered_df.dropna(subset=['created_dt'])
trend_df = trend_df[trend_df['created_dt'] >= '2025-01-01']

if len(trend_df) > 0:
    monthly_trend = trend_df.groupby('month').agg({
        'id': 'count',
        'salary_mid': 'mean'
    }).reset_index()
    monthly_trend['month'] = monthly_trend['month'].astype(str)

    st.subheader("Monthly job posting volume and average salary trend in 2025")
    fig4, ax1 = plt.subplots(figsize=(12, 6))

    color1 = '#2E86AB'
    ax1.set_xlabel('month', fontsize=12)
    ax1.set_ylabel('job posting volume', color=color1, fontsize=12)
    ax1.plot(
        monthly_trend['month'],
        monthly_trend['id'],
        color=color1,
        marker='o',
        linewidth=2,
        label='job posting volume'
    )
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(axis='y', alpha=0.3)

    ax2 = ax1.twinx()
    color2 = '#A23B72'
    ax2.set_ylabel('average salary (GBP/year)', color=color2, fontsize=12)
    ax2.plot(
        monthly_trend['month'],
        monthly_trend['salary_mid'],
        color=color2,
        marker='s',
        linewidth=2,
        label='average salary'
    )
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title("Monthly Trends in the UK Labour Market in 2025 (Number of Jobs and Average Wages)", fontsize=14, fontweight='bold')
    st.pyplot(fig4)

    st.write("### Conclusion 3: Time Trend Characteristics")
    max_jobs_month = monthly_trend.loc[monthly_trend['id'].idxmax(), 'month']
    max_salary_month = monthly_trend.loc[monthly_trend['salary_mid'].idxmax(), 'month']
    st.markdown(f"""
    - **Peak Job Month**: {max_jobs_month} with a total of {monthly_trend['id'].max()} job postings
    - **Peak Salary Month**: {max_salary_month} with an average salary of £{monthly_trend['salary_mid'].max():.0f}
    - **Trend Changes**: {(monthly_trend['id'].iloc[-1] - monthly_trend['id'].iloc[0])/monthly_trend['id'].iloc[0]*100:.1f}% change in job postings and {(monthly_trend['salary_mid'].iloc[-1] - monthly_trend['salary_mid'].iloc[0])/monthly_trend['salary_mid'].iloc[0]*100:.1f}% change in average salary
    """)
else:
    st.warning("There is no valid time data under the current filtering conditions, and the trend chart cannot be displayed")