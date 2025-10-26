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


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(
    page_title="英国劳动力市场数据分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# 1. 数据加载与预处理
# ----------------------
@st.cache_data  # 缓存数据以提高性能
def load_and_preprocess_data():
    # 加载数据（请替换为你的CSV文件路径）
    df = pd.read_csv("dfall_clean.csv")
    
    # 数据预处理
    # 1.1 处理薪资异常值（移除0或过小的薪资）
    df = df[(df['salary_mid'] > 1000) | (df['salary_mid'].isna())]
    
    # 1.2 处理日期格式
    df['created_dt'] = pd.to_datetime(df['created_dt'], errors='coerce')
    df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
    df['month'] = df['created_dt'].dt.to_period('M')  # 按月分组
    
    # 1.3 提取地区信息（从location_area中提取一级地区）
    df['region'] = df['location_area'].str.extract(r'UK > ([^ >]+)')
    df['region'] = df['region'].fillna('Unknown')
    
    # 1.4 处理岗位类型缺失值
    df['contract_type'] = df['contract_type'].fillna('Unknown')
    df['contract_time'] = df['contract_time'].fillna('Unknown')
    
    # 1.5 过滤主要岗位类别（选择样本数>20的类别）
    category_counts = df['category_label'].value_counts()
    main_categories = category_counts[category_counts > 20].index.tolist()
    df['main_category'] = df['category_label'].where(df['category_label'].isin(main_categories), 'Other')
    
    return df

# 加载数据
df = load_and_preprocess_data()

# ----------------------
# 2. 侧边栏过滤组件（满足"至少两种过滤组件"要求）
# ----------------------
st.sidebar.header("数据过滤条件")

# 2.1 薪资范围滑块（过滤组件1：滑块）
salary_min, salary_max = st.sidebar.slider(
    "薪资范围（年度，英镑）",
    min_value=int(df['salary_mid'].min()),
    max_value=int(df['salary_mid'].max()),
    value=(20000, 80000),
    step=5000
)

# 2.2 岗位类别多选框（过滤组件2：多选）
selected_categories = st.sidebar.multiselect(
    "岗位类别",
    options=df['main_category'].unique(),
    default=['IT Jobs', 'Teaching Jobs', 'Healthcare & Nursing Jobs', 'Logistics & Warehouse Jobs']
)

# 2.3 工作时间类型单选
contract_time = st.sidebar.radio(
    "工作时间类型",
    options=['All', 'full_time', 'part_time'],
    index=0
)

# 应用过滤条件
filtered_df = df[
    (df['salary_mid'] >= salary_min) & 
    (df['salary_mid'] <= salary_max) & 
    (df['main_category'].isin(selected_categories))
]

if contract_time != 'All':
    filtered_df = filtered_df[filtered_df['contract_time'] == contract_time]

# ----------------------
# 3. 主页面内容
# ----------------------
st.title("📊 英国劳动力市场数据分析报告")
st.subheader(f"当前过滤条件：薪资{salary_min}-{salary_max}英镑 | 岗位类别{selected_categories} | 工作时间{contract_time}")
st.divider()

# 3.1 数据概览
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("总岗位数", f"{len(filtered_df):,}")
with col2:
    st.metric("平均薪资", f"£{filtered_df['salary_mid'].mean():.0f}")
with col3:
    st.metric("涉及地区数", filtered_df['region'].nunique())
with col4:
    st.metric("数据时间范围", f"{filtered_df['created_date'].min().strftime('%Y-%m')}至{filtered_df['created_date'].max().strftime('%Y-%m')}")

st.divider()

# ----------------------
# 4. 核心分析模块（满足"至少两个查询问题"要求）
# ----------------------

# ======================
# 问题1：不同岗位类别的薪资分布与对比
# （包含图表1：箱线图；图表2：条形图）
# ======================
st.header("🔍 问题1：不同岗位类别的薪资水平对比")

# 4.1 图表1：岗位类别薪资箱线图（展示薪资分布离散度）
st.subheader("各岗位类别薪资分布箱线图")
fig1, ax1 = plt.subplots(figsize=(12, 6))
filtered_df_box = filtered_df[filtered_df['main_category'].isin(selected_categories)]

# 按平均薪资排序岗位类别
category_avg_salary = filtered_df_box.groupby('main_category')['salary_mid'].mean().sort_values(ascending=False)
ordered_categories = category_avg_salary.index.tolist()

# 绘制箱线图
box_plot = ax1.boxplot(
    [filtered_df_box[filtered_df_box['main_category'] == cat]['salary_mid'] for cat in ordered_categories],
    labels=ordered_categories,
    patch_artist=True,
    showfliers=False  # 隐藏异常值以提高可读性
)

# 设置颜色
colors = plt.cm.Set3(np.linspace(0, 1, len(ordered_categories)))
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

ax1.set_xlabel("岗位类别", fontsize=12)
ax1.set_ylabel("薪资（英镑/年）", fontsize=12)
ax1.set_title("不同岗位类别的薪资分布（中位数、四分位数）", fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig1)

# 4.2 图表2：岗位类别平均薪资条形图（展示平均水平）
st.subheader("各岗位类别平均薪资条形图")
fig2, ax2 = plt.subplots(figsize=(12, 6))

# 计算平均薪资并排序
avg_salary_by_category = filtered_df.groupby('main_category')['salary_mid'].agg(['mean', 'count']).sort_values('mean', ascending=False)
avg_salary_by_category = avg_salary_by_category[avg_salary_by_category['count'] >= 5]  # 过滤样本数<5的类别

# 绘制条形图
bars = ax2.bar(
    avg_salary_by_category.index,
    avg_salary_by_category['mean'],
    color=plt.cm.viridis(np.linspace(0, 1, len(avg_salary_by_category))),
    alpha=0.8
)

# 在条形图上添加数值标签和样本数
for i, (idx, row) in enumerate(avg_salary_by_category.iterrows()):
    ax2.text(
        i, row['mean'] + 1000,
        f"£{row['mean']:.0f}\n(n={row['count']})",
        ha='center', va='bottom', fontsize=10
    )

ax2.set_xlabel("岗位类别", fontsize=12)
ax2.set_ylabel("平均薪资（英镑/年）", fontsize=12)
ax2.set_title("不同岗位类别的平均薪资对比（样本数≥5）", fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig2)

# 4.3 问题1结论
st.write("### 结论1：岗位类别薪资特征")
st.markdown(f"""
- **高薪岗位Top3**：{avg_salary_by_category.head(3).index.tolist()}，平均薪资均超过£{avg_salary_by_category.head(3)['mean'].min():.0f}
- **薪资差异**：最高薪类别与最低薪类别的薪资差距约£{avg_salary_by_category['mean'].max() - avg_salary_by_category['mean'].min():.0f}
- **稳定性**：{ordered_categories[0]}岗位薪资分布最集中（箱线图高度最窄），{ordered_categories[-1]}岗位薪资波动最大
""")

st.divider()

# ======================
# 问题2：岗位地理分布与薪资的关系
# （包含图表3：地图；图表4：地区薪资直方图）
# ======================
st.header("🔍 问题2：岗位地理分布与地区薪资水平")

# 4.1 图表3：岗位地理分布图（散点地图）
st.subheader("岗位地理分布地图（基于经纬度）")
# 过滤有经纬度数据的记录
map_df = filtered_df.dropna(subset=['latitude', 'longitude'])

if len(map_df) > 0:
    # 创建地图中心（英国地理中心）
    uk_center = [55.3781, -3.4360]
    m = folium.Map(location=uk_center, zoom_start=6)
    
    # 按岗位类别设置颜色
    category_color_map = {cat: plt.cm.tab10(i) for i, cat in enumerate(selected_categories)}
    
    # 添加岗位标记
    for idx, row in map_df.iterrows():
        # 弹窗内容
        popup_content = f"""
        <b>岗位名称</b>: {row['title']}<br>
        <b>公司</b>: {row['company_name']}<br>
        <b>类别</b>: {row['main_category']}<br>
        <b>薪资</b>: £{row['salary_mid']:.0f}/年<br>
        <b>地区</b>: {row['location_display']}
        """
        
        # 添加圆形标记（薪资越高，半径越大）
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
    
    # 添加图例
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 1px 5px rgba(0,0,0,0.2);">
    <b>岗位类别图例</b><br>
    """
    for cat, color in category_color_map.items():
        color_hex = mpl.colors.to_hex(color)
        legend_html += f"<span style='display: inline-block; width: 12px; height: 12px; background: {color_hex}; margin-right: 5px;'></span>{cat}<br>"
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 在Streamlit中显示地图
    st_folium(m, width=1200, height=600)
else:
    st.warning("当前过滤条件下无有效经纬度数据，无法显示地图")

# 4.2 图表4：地区薪资分布直方图（满足"三种图表类型"要求）
st.subheader("主要地区薪资分布直方图")
fig3, ax3 = plt.subplots(figsize=(12, 6))

# 选择样本数>10的地区
region_counts = filtered_df['region'].value_counts()
main_regions = region_counts[region_counts > 10].index.tolist()
region_salary_df = filtered_df[filtered_df['region'].isin(main_regions)]

# 绘制多地区薪资直方图
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

ax3.set_xlabel("薪资（英镑/年）", fontsize=12)
ax3.set_ylabel("岗位数量", fontsize=12)
ax3.set_title("主要地区的薪资分布频率", fontsize=14, fontweight='bold')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(axis='y', alpha=0.3)
st.pyplot(fig3)

# 4.3 地区薪资统计表格
st.subheader("主要地区薪资统计")
region_salary_stats = filtered_df.groupby('region')['salary_mid'].agg([
    'count', 'mean', 'median', 'std'
]).round(0)
region_salary_stats = region_salary_stats[region_salary_stats['count'] >= 5].sort_values('mean', ascending=False)
region_salary_stats.columns = ['岗位数量', '平均薪资', '薪资中位数', '薪资标准差']
st.dataframe(region_salary_stats)

# 4.4 问题2结论
st.write("### 结论2：地理分布与薪资特征")
if len(region_salary_stats) > 0:
    st.markdown(f"""
    - **高薪地区Top3**：{region_salary_stats.head(3).index.tolist()}，平均薪资均超过£{region_salary_stats.head(3)['平均薪资'].min():.0f}
    - **岗位密集区**：{region_counts.index[0]}拥有最多岗位（{region_counts.iloc[0]}个），占总岗位数的{region_counts.iloc[0]/len(filtered_df)*100:.1f}%
    - **薪资稳定性**：{region_salary_stats['薪资标准差'].idxmin()}地区薪资最稳定（标准差£{region_salary_stats['薪资标准差'].min():.0f}）
    """)

st.divider()

# ======================
# 问题3：劳动力市场时间趋势（岗位发布量与薪资变化）
# （包含图表5：双轴折线图）
# ======================
st.header("🔍 问题3：劳动力市场时间趋势分析")

# 过滤有日期数据的记录
trend_df = filtered_df.dropna(subset=['created_dt'])
trend_df = trend_df[trend_df['created_dt'] >= '2025-01-01']  # 只分析2025年数据

if len(trend_df) > 0:
    # 按月份聚合数据
    monthly_trend = trend_df.groupby('month').agg({
        'id': 'count',  # 岗位数量
        'salary_mid': 'mean'  # 平均薪资
    }).reset_index()
    monthly_trend['month'] = monthly_trend['month'].astype(str)  # 转换为字符串便于显示

    # 图表5：双轴折线图（岗位数量+平均薪资）
    st.subheader("2025年月度岗位发布量与平均薪资趋势")
    fig4, ax1 = plt.subplots(figsize=(12, 6))

    # 轴1：岗位数量
    color1 = '#2E86AB'
    ax1.set_xlabel('月份', fontsize=12)
    ax1.set_ylabel('岗位发布数量', color=color1, fontsize=12)
    ax1.plot(
        monthly_trend['month'],
        monthly_trend['id'],
        color=color1,
        marker='o',
        linewidth=2,
        label='岗位数量'
    )
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(axis='y', alpha=0.3)

    # 轴2：平均薪资
    ax2 = ax1.twinx()
    color2 = '#A23B72'
    ax2.set_ylabel('平均薪资（英镑/年）', color=color2, fontsize=12)
    ax2.plot(
        monthly_trend['month'],
        monthly_trend['salary_mid'],
        color=color2,
        marker='s',
        linewidth=2,
        label='平均薪资'
    )
    ax2.tick_params(axis='y', labelcolor=color2)

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title("2025年英国劳动力市场月度趋势（岗位数量与平均薪资）", fontsize=14, fontweight='bold')
    st.pyplot(fig4)

    # 时间趋势结论
    st.write("### 结论3：时间趋势特征")
    max_jobs_month = monthly_trend.loc[monthly_trend['id'].idxmax(), 'month']
    max_salary_month = monthly_trend.loc[monthly_trend['salary_mid'].idxmax(), 'month']
    st.markdown(f"""
    - **岗位高峰月**：{max_jobs_month}，共发布{monthly_trend['id'].max()}个岗位
    - **薪资高峰月**：{max_salary_month}，平均薪资£{monthly_trend['salary_mid'].max():.0f}
    - **趋势变化**：{(monthly_trend['id'].iloc[-1] - monthly_trend['id'].iloc[0])/monthly_trend['id'].iloc[0]*100:.1f}%的岗位数量变化率，{(monthly_trend['salary_mid'].iloc[-1] - monthly_trend['salary_mid'].iloc[0])/monthly_trend['salary_mid'].iloc[0]*100:.1f}%的平均薪资变化率
    """)
else:
    st.warning("当前过滤条件下无有效时间数据，无法显示趋势图")

# ----------------------
# 5. 数据导出功能
# ----------------------
st.divider()
st.header("💾 数据导出")
st.write("可导出当前过滤条件下的原始数据用于进一步分析")

# 准备导出数据
export_df = filtered_df[['id', 'title', 'company_name', 'main_category', 'region', 
                        'location_display', 'contract_type', 'contract_time', 
                        'salary_min', 'salary_max', 'salary_mid', 'created_date']]

# 导出按钮
csv_data = export_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="导出CSV文件",
    data=csv_data,
    file_name=f"uk_labour_market_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

# ----------------------
# 6. 项目说明
# ----------------------
st.divider()
st.header("📋 项目说明")
st.markdown("""
### 功能亮点
1. **多维度分析**：覆盖岗位类别薪资、地理分布、时间趋势三大核心维度
2. **交互式体验**：提供薪资滑块、岗位类别多选、工作时间单选三种过滤组件
3. **多样化图表**：包含箱线图、条形图、直方图、地图、双轴折线图五种图表类型
4. **数据导出**：支持导出过滤后的数据用于离线分析

### 数据来源
- 数据集：英国招聘平台Adzuna的岗位数据（dfall_clean.csv）
- 核心字段：岗位信息、公司信息、地理信息、薪资信息、发布时间

### 技术栈
- 数据处理：Pandas、NumPy
- 可视化：Matplotlib、Folium、Seaborn
- 应用框架：Streamlit
""")
