# -*- coding: utf-8 -*-
"""
湖南省招聘平台差异化策略研究 —— K-Means聚类 + Apriori关联规则
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 阶段一：数据预处理
# ===========================

df = pd.read_excel(r'D:\LLY\毕业实训test\zhaopin-hunan_20260710152827.xlsx', engine='openpyxl')
print(f"原始数据: {df.shape}")

# --- 列名映射 ---
COL = {
    'company': '企业名称', 'job': '招聘岗位', 'city': '工作城市', 'district': '工作区域',
    'min_salary': '最低月薪', 'max_salary': '最高月薪', 'desc': '职位描述',
    'edu': '学历要求', 'exp': '要求经验', 'count': '招聘人数',
    'type': '招聘类别', 'industry_raw': '初级分类', 'platform': '来源平台',
    'company_addr': '公司地点', 'work_addr': '工作地点',
    'pub_date': '招聘发布日期', 'end_date': '招聘结束日期',
    'pub_year': '招聘发布年份', 'end_year': '招聘结束年份', 'source': '来源'
}

# 1. 缺失值处理
df = df.dropna(subset=[COL['type']])  # 删除招聘类型为NaN的行
df = df[~((df[COL['min_salary']].isna()) & (df[COL['max_salary']].isna()))]  # 删除月薪均为空的
df = df[~((df[COL['min_salary']] == 0) & (df[COL['max_salary']] == 0))]  # 删除月薪均为0的
# 平台NaN标记为"未知"
df[COL['platform']] = df[COL['platform']].fillna('未知')
# 行业NaN填充为"其他行业"
df[COL['industry_raw']] = df[COL['industry_raw']].fillna('其他行业')

print(f"清洗后: {df.shape}")

# 2. 薪资字段：平均月薪
df['avg_salary'] = (df[COL['min_salary']].fillna(df[COL['max_salary']]) +
                    df[COL['max_salary']].fillna(df[COL['min_salary']])) / 2

# 3. 行业分类：从原始行业字段清洗并合并为6大类
INDUSTRY_MAP = {
    '电子/半导体/集成电路': 'IT/互联网', '互联网': 'IT/互联网', '计算机软件': 'IT/互联网',
    '通信/网络设备': 'IT/互联网', 'IT服务': 'IT/互联网', '计算机硬件': 'IT/互联网',
    '人工智能': 'IT/互联网', '云计算/大数据': 'IT/互联网', '网络/信息安全': 'IT/互联网',
    '物联网': 'IT/互联网', '智能硬件': 'IT/互联网', '游戏': 'IT/互联网',
    '电子商务': 'IT/互联网', '新媒体': 'IT/互联网', '软件/互联网开发/系统集成': 'IT/互联网',
    'IT/互联网': 'IT/互联网', '电子/通信': 'IT/互联网', '电子/电器/半导体/仪器仪表': 'IT/互联网',
    '在线教育': 'IT/互联网', '在线音乐/视频/阅读': 'IT/互联网', '在线医疗': 'IT/互联网',
    '社交网络': 'IT/互联网', '区块链': 'IT/互联网', '消费电子产品': 'IT/互联网',
    '运营商/增值服务': 'IT/互联网', '光电子行业': 'IT/互联网',
    '汽车智能互联': 'IT/互联网', '机器人': 'IT/互联网',
    '电气机械/电力设备': '制造业', '专用设备制造': '制造业', '仪器仪表制造': '制造业',
    '医药制造': '制造业', '医疗设备/器械': '制造业', '金属制品业': '制造业',
    '电子设备制造': '制造业', '通用设备制造': '制造业', '汽车零部件': '制造业',
    '化学原料/化学制品': '制造业', '化工': '制造业', '建材': '制造业',
    '非金属矿物制品业': '制造业', '钢铁/有色金属冶炼及加工': '制造业',
    '橡胶和塑料制品': '制造业', '家具制造': '制造业', '纺织业/服饰产品加工制造': '制造业',
    '化学纤维制造业': '制造业', '日化产品制造': '制造业', '文体/办公设备制造': '制造业',
    '食品/饮料': '制造业', '农副产品加工制造': '制造业', '印刷/包装/造纸': '制造业',
    '生产制造/造纸印刷/服装服饰': '制造业', '烟草/酒业': '制造业',
    '耐用消费品': '制造业', '快速消费品': '制造业',
    '工业自动化': '制造业', '军工制造': '制造业',
    '汽车研发/制造': '制造业', '新能源汽车': '制造业', '船舶/航空/航天/火车制造': '制造业',
    '教育': '教育', '培训/辅导服务': '教育', '学校/学历教育': '教育', '学术/科研': '教育',
    '金融': '金融', '证券/期货': '金融', '银行': '金融', '投资/融资': '金融',
    '保险': '金融', '汽车金融': '金融', '基金': '金融',
    '房地产开发': '建筑/地产', '工程施工': '建筑/地产', '建筑设备安装': '建筑/地产',
    '装饰装修': '建筑/地产', '建筑设计': '建筑/地产', '建筑工程检测': '建筑/地产',
    '房地产/建筑': '建筑/地产', '房地产/工程/建筑': '建筑/地产', '房地产中介': '建筑/地产',
    '物业管理': '建筑/地产', '土木/建筑/装修/市政工程': '建筑/地产',
    '新能源': '能源/环保', '环保': '能源/环保', '电力/水利/热力/燃气': '能源/环保',
    '石油/石化': '能源/环保', '矿产/采掘': '能源/环保', '新材料': '能源/环保',
    '能源/采掘/化工/环保': '能源/环保',
    '医疗服务': '医疗/健康', '卫生服务': '医疗/健康', '医药批发/零售': '医疗/健康',
    '医院': '医疗/健康', '医美/健康服务': '医疗/健康', '医疗检测': '医疗/健康',
    '生物工程': '医疗/健康', 'IVD': '医疗/健康', '生物/制药/医疗器械': '医疗/健康',
    '制药/医疗': '医疗/健康',
    '人力资源服务': '服务业', '咨询服务': '服务业', '企业服务': '服务业',
    '检测/认证': '服务业', '专业技术服务': '服务业', '工程技术与设计服务': '服务业',
    '货运/物流/仓储': '服务业', '贸易/进出口': '服务业', '零售/批发': '服务业',
    '餐饮服务': '服务业', '住宿/餐饮': '服务业', '旅游服务': '服务业',
    '邮政/快递': '服务业', '租赁服务': '服务业', '销售/客服/技术支持': '服务业',
    '人事/行政/财务/法务': '服务业', '财务/审计/税务': '服务业',
    '咨询/法律/翻译/商标/专利': '服务业', '财务/人力资源/行政': '服务业',
    '居民服务': '服务业', '家政服务': '服务业', '行政/后勤/文秘': '服务业',
    '专利/商标/知识产权': '服务业', '商业代理服务': '服务业', '科学技术推广': '服务业',
    '客运服务': '服务业', '酒店/民宿': '服务业', '文化艺术/娱乐': '服务业',
    '广播/影视': '服务业', '广告/营销': '服务业', '新闻/出版': '服务业',
    '美发/美容/保健': '服务业', '休闲/娱乐': '服务业', '体育': '服务业',
    '政府/公共事业': '服务业', '景区/商业/市场等综合管理': '服务业',
    '土地与公共设施管理': '服务业', '火车站/港口/汽车站/路政': '服务业',
    '租赁/拍卖/典当/担保': '服务业',
}

df['industry'] = df[COL['industry_raw']].map(INDUSTRY_MAP).fillna('其他')

# 4. 学历量化
EDU_MAP = {
    '学历不限': 1, '不限': 1, '初中及以下': 1, '高中': 1,
    '中专': 2, '中技': 2, '技校': 2, '中专/中技': 2,
    '大专': 3,
    '本科': 4, 'EMBA': 4, 'MBA/EMBA': 4, '其他': 3,
    '硕士': 5,
    '博士': 6
}
df['edu_level'] = df[COL['edu']].map(EDU_MAP).fillna(2)

# 5. 经验量化
EXP_MAP = {
    '经验不限': 0,
    '不限': 0,
    '1年以下': 0.5,
    '1年': 1.0,
    '1-3年': 1.5,
    '2年': 2.0,
    '2年以上': 2.5,
    '3年': 3.0,
    '3-5年': 3.5,
    '3年以上': 4.0,
    '5-7年': 5.5,
    '5-10年': 7.0,
    '5年以上': 7.0,
    '10年以上': 12.0,
}
df['exp_value'] = df[COL['exp']].map(EXP_MAP).fillna(0)

# 6. 城市归类：保留Top 6 + 长沙，其余归为"其他城市"
city_top = df[COL['city']].value_counts().head(7).index.tolist()
# 确保长沙在列
if '长沙' not in city_top:
    city_top = ['长沙'] + city_top[:6]
else:
    city_top = city_top[:7]
df['city_group'] = df[COL['city']].apply(lambda x: x if x in city_top else '其他城市')

# 7. 薪资分箱（等频分箱4档）
df['salary_bin_raw'] = pd.qcut(df['avg_salary'], q=4, labels=False, duplicates='drop')
salary_labels = {0: '低薪(<=5k)', 1: '中薪(5-10k)', 2: '高薪(10-20k)', 3: '超高薪(>20k)'}
# 实际分箱值可能不同，重新映射
unique_bins = sorted(df['salary_bin_raw'].dropna().unique())
n_bins = len(unique_bins)
if n_bins == 4:
    bin_labels = ['低薪(<=5k)', '中薪(5-10k)', '高薪(10-20k)', '超高薪(>20k)']
elif n_bins == 3:
    bin_labels = ['低薪(<8k)', '中薪(8-15k)', '高薪(>15k)']
else:
    bin_labels = [f'薪资档{i+1}' for i in range(n_bins)]
bin_map = {unique_bins[i]: bin_labels[i] for i in range(n_bins)}
df['salary_bin'] = df['salary_bin_raw'].map(bin_map)

# 8. 平台处理
platform_counts = df[COL['platform']].value_counts()
top_platforms = set()
for p, c in platform_counts.items():
    if c > 100:
        top_platforms.add(p)
# 标记: 智联招聘是主要来源平台，来源平台列大部分为"马克数据网"(数据供应商)或NaN
df['platform_group'] = df[COL['platform']].apply(
    lambda x: x if x in top_platforms else '其他平台'
)

print(f"\n预处理完成。有效记录: {len(df)}")
print(f"平台分布: {df['platform_group'].value_counts().to_dict()}")
print(f"城市分布: {df['city_group'].value_counts().to_dict()}")
print(f"行业分布: {df['industry'].value_counts().to_dict()}")
print(f"薪资分箱: {df['salary_bin'].value_counts().to_dict()}")

# ===========================
# 阶段二：K-Means聚类
# ===========================

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

cluster_cols = ['exp_value', 'edu_level', 'avg_salary']
cluster_data = df[cluster_cols].dropna().copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_data)

# 肘部法则 + 轮廓系数
inertias = []
silhouettes = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)

best_k_idx = np.argmax(silhouettes)
best_k = list(k_range)[best_k_idx]
print(f"\n最佳K值: {best_k} (轮廓系数={silhouettes[best_k_idx]:.4f})")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 命名聚类
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=cluster_cols
)
cluster_centers['size'] = df['cluster'].value_counts().sort_index()
cluster_centers['pct'] = (df['cluster'].value_counts(normalize=True).sort_index() * 100).round(1)

# 根据中心点命名
def name_cluster(row):
    exp, edu, sal = row['exp_value'], row['edu_level'], row['avg_salary']
    if sal > 20000 and edu >= 4.5:
        return '高端管理/技术岗'
    elif sal > 12000 and edu >= 3.5:
        return '资深专业岗'
    elif sal > 6000 and edu >= 2.5:
        return '中级技术岗'
    else:
        return '初级基础岗'

cluster_centers['name'] = cluster_centers.apply(name_cluster, axis=1)
# 如果有重复名称，加上编号
name_counts = {}
final_names = []
for n in cluster_centers['name']:
    if n not in name_counts:
        name_counts[n] = 0
    name_counts[n] += 1
    if name_counts[n] > 1:
        final_names.append(f"{n}-{name_counts[n]}")
    else:
        final_names.append(n)
cluster_centers['name'] = final_names

cluster_name_map = dict(zip(cluster_centers.index, cluster_centers['name']))
df['cluster_name'] = df['cluster'].map(cluster_name_map)

print("\n=== 聚类中心 ===")
print(cluster_centers[['name', 'exp_value', 'edu_level', 'avg_salary', 'size', 'pct']].to_string())

# 肘部法则+轮廓系数图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, 'bo-')
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].axvline(x=best_k, color='r', linestyle='--', alpha=0.7)
axes[1].plot(list(k_range), silhouettes, 'go-')
axes[1].set_xlabel('K')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Coefficient')
axes[1].axvline(x=best_k, color='r', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'D:\LLY\毕业实训test\fig01_kmeans_elbow.png', dpi=150)
plt.close()
print("-> fig01_kmeans_elbow.png 已保存")

# 聚类3D散点图（投影到2D）
fig, ax = plt.subplots(figsize=(10, 6))
sample = df.sample(min(5000, len(df)), random_state=42)
scatter = ax.scatter(sample['exp_value'], sample['avg_salary'],
                     c=sample['cluster'], cmap='viridis', alpha=0.5, s=10)
centroids = cluster_centers.reset_index(drop=True)
ax.scatter(centroids['exp_value'], centroids['avg_salary'],
           c='red', marker='X', s=200, edgecolors='black', linewidths=1.5)
for i, row in centroids.iterrows():
    ax.annotate(row['name'], (row['exp_value'], row['avg_salary']),
                fontsize=9, fontweight='bold', ha='center',
                xytext=(0, -15), textcoords='offset points')
ax.set_xlabel('Experience')
ax.set_ylabel('Avg Salary')
ax.set_title('K-Means Clusters: Experience vs Salary')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig(r'D:\LLY\毕业实训test\fig02_cluster_scatter.png', dpi=150)
plt.close()
print("-> fig02_cluster_scatter.png 已保存")

# ===========================
# 阶段三：Apriori关联规则挖掘
# ===========================

from mlxtend.frequent_patterns import apriori, association_rules

# 构建离散化标签（平台 × 城市 × 行业 × 薪资 × 岗位层级）
apriori_df = df[['platform_group', 'city_group', 'industry', 'salary_bin', 'cluster_name']].copy()
apriori_df.columns = ['平台', '城市', '行业', '薪资', '岗位层级']

# 添加前缀
for col in apriori_df.columns:
    apriori_df[col] = col + '=' + apriori_df[col].astype(str)

# 独热编码
onehot = pd.get_dummies(apriori_df)

print(f"\n独热编码维度: {onehot.shape}")

# 频繁项集挖掘
min_support = 0.02  # 约1851条
frequent_itemsets = apriori(onehot, min_support=min_support, use_colnames=True, max_len=5)
print(f"频繁项集数: {len(frequent_itemsets)}")

# 关联规则
min_confidence = 0.5
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
# 只保留lift > 1的
rules = rules[rules['lift'] > 1]

# 按规则后件为"平台"的筛选出来
def rule_has_platform_consequent(consequents):
    for item in consequents:
        if item.startswith('平台='):
            return True
    return False

rules_with_platform = rules[rules['consequents'].apply(rule_has_platform_consequent)]
rules_other = rules[~rules['consequents'].apply(rule_has_platform_consequent)]

# 合并并按lift排序
rules = pd.concat([rules_with_platform, rules_other]).head(200)

print(f"关联规则总数: {len(rules)} (其中后件为平台的: {len(rules_with_platform)})")

# 保存规则
rules_save = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
rules_save['antecedents'] = rules_save['antecedents'].apply(lambda x: ', '.join(sorted(x)))
rules_save['consequents'] = rules_save['consequents'].apply(lambda x: ', '.join(sorted(x)))
rules_save = rules_save.sort_values('lift', ascending=False)
rules_save.to_csv(r'D:\LLY\毕业实训test\association_rules.csv', index=False, encoding='utf-8-sig')

# 打印Top 20规则
pd.set_option('display.max_colwidth', 80)
print("\n=== Top 15 关联规则 (按Lift降序) ===")
for i, (_, r) in enumerate(rules_save.head(15).iterrows()):
    print(f"  [{r['antecedents']}] => [{r['consequents']}] "
          f"sup={r['support']:.4f} conf={r['confidence']:.3f} lift={r['lift']:.2f}")

# ===========================
# 阶段四：综合可视化
# ===========================

# 图3: 平台-城市-行业关联热力图
pivot = df.pivot_table(index='city_group', columns='industry',
                        values='avg_salary', aggfunc='count', fill_value=0)
# 只保留主要城市和行业
top_cities = df['city_group'].value_counts().head(8).index
top_industries = df['industry'].value_counts().head(8).index
pivot_filtered = pivot.loc[pivot.index.isin(top_cities),
                           [c for c in top_industries if c in pivot.columns]]
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pivot_filtered, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': '岗位数'})
ax.set_title('城市×行业岗位数热力图')
ax.set_xlabel('行业')
ax.set_ylabel('城市')
plt.tight_layout()
plt.savefig(r'D:\LLY\毕业实训test\fig03_city_industry_heatmap.png', dpi=150)
plt.close()
print("-> fig03_city_industry_heatmap.png 已保存")

# 图4: 薪资箱线图 (按岗位层级)
fig, ax = plt.subplots(figsize=(10, 5))
salary_sample = df[df['avg_salary'] < 50000].sample(min(8000, len(df)), random_state=42)
sns.boxplot(data=salary_sample, x='cluster_name', y='avg_salary', palette='Set2', ax=ax)
ax.set_title('各岗位层级薪资分布')
ax.set_xlabel('岗位层级')
ax.set_ylabel('平均月薪(元)')
ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig(r'D:\LLY\毕业实训test\fig04_salary_boxplot.png', dpi=150)
plt.close()
print("-> fig04_salary_boxplot.png 已保存")

# 图5: 各城市岗位层级堆叠柱状图
ct_cross = pd.crosstab(df['city_group'], df['cluster_name'])
ct_cross_pct = ct_cross.div(ct_cross.sum(axis=1), axis=0) * 100
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ct_cross.loc[ct_cross.sum(axis=1).sort_values(ascending=False).head(10).index].plot(
    kind='bar', stacked=True, ax=axes[0], colormap='Set2')
axes[0].set_title('各城市岗位层级分布(绝对数)')
axes[0].set_xlabel('城市')
axes[0].set_ylabel('岗位数')
axes[0].tick_params(axis='x', rotation=30)
ct_cross_pct.loc[ct_cross_pct.sum(axis=1).sort_values(ascending=False).head(10).index].plot(
    kind='bar', stacked=True, ax=axes[1], colormap='Set2')
axes[1].set_title('各城市岗位层级分布(占比%)')
axes[1].set_xlabel('城市')
axes[1].set_ylabel('占比(%)')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(r'D:\LLY\毕业实训test\fig05_city_cluster_bar.png', dpi=150)
plt.close()
print("-> fig05_city_cluster_bar.png 已保存")

# 图6: 行业-岗位层级热力图
pivot2 = df.pivot_table(index='industry', columns='cluster_name',
                         values='avg_salary', aggfunc='count', fill_value=0)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pivot2, annot=True, fmt='.0f', cmap='YlGnBu', ax=ax,
            linewidths=0.5, cbar_kws={'label': '岗位数'})
ax.set_title('行业×岗位层级分布热力图')
ax.set_xlabel('岗位层级')
ax.set_ylabel('行业')
plt.tight_layout()
plt.savefig(r'D:\LLY\毕业实训test\fig06_industry_cluster_heatmap.png', dpi=150)
plt.close()
print("-> fig06_industry_cluster_heatmap.png 已保存")

# 图7: 关联规则散点图 & 网络图
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(rules['support'], rules['confidence'], c=rules['lift'],
                s=rules['lift'] * 20, cmap='Reds', alpha=0.7, edgecolors='gray', linewidths=0.5)
top5 = rules.nlargest(5, 'lift')
for _, r in top5.iterrows():
    ant_str = ', '.join(sorted(list(r['antecedents'])))
    con_str = ', '.join(sorted(list(r['consequents'])))
    ax.annotate(f"{ant_str[:30]}...", (r['support'], r['confidence']),
                fontsize=7, alpha=0.8,
                xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('Support')
ax.set_ylabel('Confidence')
ax.set_title('关联规则散点图 (颜色=Lift, 大小=Lift)')
plt.colorbar(sc, label='Lift')
plt.tight_layout()
plt.savefig(r'D:\LLY\毕业实训test\fig07_rules_scatter.png', dpi=150)
plt.close()
print("-> fig07_rules_scatter.png 已保存")

# 图8: 关联规则网络图
try:
    import networkx as nx
    top_rules_net = rules.nlargest(30, 'lift')
    G = nx.DiGraph()
    for _, r in top_rules_net.iterrows():
        for ant in r['antecedents']:
            for con in r['consequents']:
                if not G.has_edge(ant, con):
                    G.add_edge(ant, con, weight=r['lift'], conf=r['confidence'])
    fig, ax = plt.subplots(figsize=(18, 14))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    edges = G.edges()
    weights = [G[u][v]['weight'] * 2 for u, v in edges]
    colors = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue',
                           edgecolors='gray', ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, edge_color=colors,
                           edge_cmap=plt.cm.Reds, alpha=0.6, arrows=True,
                           arrowsize=15, connectionstyle='arc3,rad=0.1', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax)
    ax.set_title('关联规则网络图 (Top 30 Rules by Lift)', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(r'D:\LLY\毕业实训test\fig08_rule_network.png', dpi=150)
    plt.close()
    print("-> fig08_rule_network.png 已保存")
except Exception as e:
    print(f"网络图生成跳过: {e}")

# ===========================
# 阶段五：汇总 & 策略建议
# ===========================

print("\n" + "=" * 60)
print("阶段五：结论与运营策略建议")
print("=" * 60)

# 核心统计
print("\n=== 基本统计 ===")
print(f"数据总量: {len(df)} 条岗位记录")
print(f"覆盖城市: {df['city_group'].nunique()} 个")
print(f"覆盖行业: {df['industry'].nunique()} 个大类")
print(f"岗位层级: {len(cluster_centers)} 类")

# Top 平台规则摘要
if len(rules_with_platform) > 0:
    print(f"\n=== 平台关联规则 ({len(rules_with_platform)}条) ===")
    for _, r in rules_with_platform.nlargest(10, 'lift').iterrows():
        ant = ', '.join(sorted(list(r['antecedents'])))
        con = ', '.join(sorted(list(r['consequents'])))
        print(f"  [{ant}] => [{con}] lift={r['lift']:.2f} conf={r['confidence']:.3f}")

# 各行业薪酬
print("\n=== 各行业平均薪资 Top 10 ===")
industry_salary = df.groupby('industry')['avg_salary'].agg(['mean', 'count']).round(0)
industry_salary = industry_salary[industry_salary['count'] > 50].sort_values('mean', ascending=False)
print(industry_salary.head(10).to_string())

# 保存最终数据
final_cols = [COL['company'], COL['job'], COL['city'], COL['district'],
              COL['min_salary'], COL['max_salary'], 'avg_salary',
              COL['edu'], 'edu_level', COL['exp'], 'exp_value',
              COL['type'], 'industry', COL['platform'], 'platform_group',
              'city_group', 'salary_bin', 'cluster', 'cluster_name']
df[final_cols].to_csv(r'D:\LLY\毕业实训test\processed_data.csv', index=False, encoding='utf-8-sig')

print(f"\n处理后的数据已保存至 processed_data.csv")
print(f"关联规则已保存至 association_rules.csv")
print("所有图表已生成完毕。")
