# -*- coding: utf-8 -*-
"""
湖南省招聘平台差异化策略研究 —— K-Means聚类 + Apriori关联规则
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("阶段一：数据预处理")
print("=" * 60)

# ======== 1. 加载 ========
t0 = time.time()
df = pd.read_excel(r'D:\LLY\毕业实训test\zhaopin-hunan_20260710152827.xlsx', engine='openpyxl')
print(f"加载完成: {df.shape}, 耗时 {time.time()-t0:.1f}s")

COL = {
    'company': '企业名称', 'job': '招聘岗位', 'city': '工作城市', 'district': '工作区域',
    'min_salary': '最低月薪', 'max_salary': '最高月薪', 'desc': '职位描述',
    'edu': '学历要求', 'exp': '要求经验', 'count': '招聘人数',
    'type': '招聘类别', 'industry_raw': '初级分类', 'platform': '来源平台',
    'company_addr': '公司地点', 'work_addr': '工作地点',
    'pub_date': '招聘发布日期', 'end_date': '招聘结束日期',
    'pub_year': '招聘发布年份', 'end_year': '招聘结束年份', 'source': '来源'
}

# ======== 2. 缺失值处理 ========
df = df.dropna(subset=[COL['type']])
df = df[~((df[COL['min_salary']].isna()) & (df[COL['max_salary']].isna()))]
df[COL['platform']] = df[COL['platform']].fillna('未知')
df[COL['industry_raw']] = df[COL['industry_raw']].fillna('其他行业')

# ======== 3. 薪资字段 ========
df['avg_salary'] = (df[COL['min_salary']].fillna(df[COL['max_salary']]) +
                    df[COL['max_salary']].fillna(df[COL['min_salary']])) / 2
df = df[df['avg_salary'] > 0].copy()
print(f"清洗后: {len(df)} 条记录")

# ======== 4. 行业分类（6大类） ========
INDUSTRY_MAP = {
    '电子/半导体/集成电路': 'IT/互联网', '互联网': 'IT/互联网', '计算机软件': 'IT/互联网',
    '通信/网络设备': 'IT/互联网', 'IT服务': 'IT/互联网', '计算机硬件': 'IT/互联网',
    '人工智能': 'IT/互联网', '云计算/大数据': 'IT/互联网', '物联网': 'IT/互联网',
    '智能硬件': 'IT/互联网', '游戏': 'IT/互联网', '电子商务': 'IT/互联网',
    '新媒体': 'IT/互联网', '软件/互联网开发/系统集成': 'IT/互联网',
    'IT/互联网': 'IT/互联网', '电子/通信': 'IT/互联网', '电子/电器/半导体/仪器仪表': 'IT/互联网',
    '在线教育': 'IT/互联网', '运营商/增值服务': 'IT/互联网', '光电子行业': 'IT/互联网',
    '汽车智能互联': 'IT/互联网', '机器人': 'IT/互联网', '社交网络': 'IT/互联网',
    '区块链': 'IT/互联网', '在线音乐/视频/阅读': 'IT/互联网', '在线医疗': 'IT/互联网',
    '消费电子产品': 'IT/互联网', '网络/信息安全': 'IT/互联网',
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
    '餐饮服务': '服务业', '旅游服务': '服务业', '邮政/快递': '服务业', '租赁服务': '服务业',
    '销售/客服/技术支持': '服务业', '人事/行政/财务/法务': '服务业', '财务/审计/税务': '服务业',
    '咨询/法律/翻译/商标/专利': '服务业', '居民服务': '服务业', '家政服务': '服务业',
    '行政/后勤/文秘': '服务业', '专利/商标/知识产权': '服务业', '商业代理服务': '服务业',
    '科学技术推广': '服务业', '客运服务': '服务业', '酒店/民宿': '服务业',
    '文化艺术/娱乐': '服务业', '广播/影视': '服务业', '广告/营销': '服务业',
    '新闻/出版': '服务业', '美发/美容/保健': '服务业', '休闲/娱乐': '服务业',
    '体育': '服务业', '政府/公共事业': '服务业', '景区/商业/市场等综合管理': '服务业',
    '土地与公共设施管理': '服务业', '火车站/港口/汽车站/路政': '服务业',
    '租赁/拍卖/典当/担保': '服务业', '财务/人力资源/行政': '服务业',
}
df['industry'] = df[COL['industry_raw']].map(INDUSTRY_MAP).fillna('其他')

# ======== 5. 学历量化 ========
EDU_MAP = {
    '学历不限': 1, '不限': 1, '初中及以下': 1, '高中': 1,
    '中专': 2, '中技': 2, '技校': 2, '中专/中技': 2,
    '大专': 3, '本科': 4, 'EMBA': 4, 'MBA/EMBA': 4, '其他': 3,
    '硕士': 5, '博士': 6
}
df['edu_level'] = df[COL['edu']].map(EDU_MAP).fillna(2)

# ======== 6. 经验量化 ========
EXP_MAP = {
    '经验不限': 0, '不限': 0, '1年以下': 0.5, '1年': 1.0,
    '1-3年': 1.5, '2年': 2.0, '2年以上': 2.5, '3年': 3.0,
    '3-5年': 3.5, '3年以上': 4.0, '5-7年': 5.5,
    '5-10年': 7.0, '5年以上': 7.0, '10年以上': 12.0,
}
df['exp_value'] = df[COL['exp']].map(EXP_MAP).fillna(0)

# ======== 7. 城市归类 ========
city_top = list(df[COL['city']].value_counts().head(7).index)
# 确保长沙在前
if '长沙' in city_top:
    city_top.remove('长沙')
    city_top = ['长沙'] + city_top[:6]
else:
    city_top = city_top[:7]
df['city_group'] = df[COL['city']].apply(lambda x: x if x in city_top else '其他城市')

# ======== 8. 薪资分箱 ========
try:
    df['salary_bin'] = pd.qcut(df['avg_salary'], q=4, labels=['低薪', '中薪', '高薪', '超高薪'], duplicates='drop')
except:
    bins = [0, 5000, 10000, 20000, 500000]
    labels = ['低薪', '中薪', '高薪', '超高薪']
    df['salary_bin'] = pd.cut(df['avg_salary'], bins=bins, labels=labels)

# ======== 9. 平台处理 ========
# 数据只有单一平台（智联招聘），保留原值并标记
df['platform_group'] = df[COL['source']].fillna('智联招聘')

print(f"\n预处理完成: {len(df)} 条")
print(f"平台: {df['platform_group'].value_counts().to_dict()}")
print(f"城市(Top7): {city_top}")
print(f"行业: {df['industry'].value_counts().to_dict()}")
print(f"薪资分箱: {df['salary_bin'].value_counts().to_dict()}")

# ==========================================
# 阶段二：K-Means聚类
# ==========================================
print("\n" + "=" * 60)
print("阶段二：K-Means聚类")
print("=" * 60)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

cluster_cols = ['exp_value', 'edu_level', 'avg_salary']
X = df[cluster_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"聚类变量维度: {X_scaled.shape}")

# 肘部法则
t0 = time.time()
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
print(f"肘部法则: {time.time()-t0:.1f}s")

# 轮廓系数（采样）
sample_idx = np.random.choice(len(X_scaled), min(8000, len(X_scaled)), replace=False)
X_sample = X_scaled[sample_idx]
silhouettes = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_sample = km.fit_predict(X_sample)
    sil = silhouette_score(X_sample, labels_sample)
    silhouettes.append(sil)
    print(f"  K={k}: silhouette={sil:.4f}")

best_k = list(range(2, 9))[np.argmax(silhouettes)]
print(f"\n最佳K值: {best_k}")

# 最终K-Means
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

centers_df = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=cluster_cols
)
centers_df.index = range(best_k)
centers_df['count'] = df['cluster'].value_counts().sort_index().values
centers_df['pct'] = (df['cluster'].value_counts(normalize=True).sort_index() * 100).values

# 命名
def name_cluster(row):
    e, d, s = row['exp_value'], row['edu_level'], row['avg_salary']
    if s > 20000: return '高端管理/技术岗'
    elif s > 12000 and d >= 3.5: return '资深专业岗'
    elif s > 5000 and d >= 2.5: return '中级技术岗'
    else: return '初级基础岗'

names = []
name_counts = {}
for _, row in centers_df.iterrows():
    n = name_cluster(row)
    name_counts[n] = name_counts.get(n, 0) + 1
    if name_counts[n] > 1:
        names.append(f"{n}-{name_counts[n]}")
    else:
        names.append(n)
centers_df['name'] = names
cluster_name_map = dict(zip(range(best_k), names))
df['cluster_name'] = df['cluster'].map(cluster_name_map)

print("\n=== 聚类中心及命名 ===")
print(centers_df[['name', 'exp_value', 'edu_level', 'avg_salary', 'count', 'pct']].to_string())

# 图1: 肘部法则+轮廓系数
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(range(2, 11)), inertias, 'bo-')
axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].axvline(x=best_k, color='r', linestyle='--', alpha=0.7)
axes[1].plot(list(range(2, 9)), silhouettes, 'go-')
axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Coefficient')
axes[1].axvline(x=best_k, color='r', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'D:\LLY\毕业实训test\fig01_elbow_silhouette.png', dpi=150)
plt.close()
print("-> fig01_elbow_silhouette.png")

# 图2: 聚类散点图
fig, ax = plt.subplots(figsize=(10, 6))
sample = df.sample(min(5000, len(df)), random_state=42)
colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for c in range(best_k):
    mask = sample['cluster'] == c
    ax.scatter(sample.loc[mask, 'exp_value'], sample.loc[mask, 'avg_salary'],
               c=[colors[c]], alpha=0.3, s=5, label=f'C{c}:{names[c]}')
ax.scatter(centers_df['exp_value'], centers_df['avg_salary'],
           c='red', marker='X', s=200, edgecolors='black', linewidths=1.5)
for i, row in centers_df.iterrows():
    ax.annotate(row['name'], (row['exp_value'], row['avg_salary']),
                fontsize=8, ha='center', xytext=(0, -15), textcoords='offset points')
ax.set_xlabel('Experience Level'); ax.set_ylabel('Average Salary (CNY)')
ax.set_title('K-Means Clusters: Experience vs Salary')
ax.legend(fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig(r'D:\LLY\毕业实训test\fig02_cluster_scatter.png', dpi=150)
plt.close()
print("-> fig02_cluster_scatter.png")

# ==========================================
# 阶段三：Apriori关联规则
# ==========================================
print("\n" + "=" * 60)
print("阶段三：Apriori关联规则挖掘")
print("=" * 60)

from mlxtend.frequent_patterns import apriori, association_rules

# 构建离散化数据框
apriori_df = df[['platform_group', 'city_group', 'industry', 'salary_bin', 'cluster_name']].copy()
apriori_df.columns = ['平台', '城市', '行业', '薪资', '岗位层级']

# 添加前缀避免列名冲突
for col in apriori_df.columns:
    apriori_df[col] = col + '=' + apriori_df[col].astype(str)

# 独热编码
t0 = time.time()
onehot = pd.get_dummies(apriori_df)
print(f"独热编码维度: {onehot.shape}")
print(f"编码耗时: {time.time()-t0:.1f}s")

# 频繁项集
min_support = 0.02
t0 = time.time()
frequent_itemsets = apriori(onehot, min_support=min_support, use_colnames=True, max_len=5)
print(f"频繁项集数: {len(frequent_itemsets)}")
print(f"Apriori耗时: {time.time()-t0:.1f}s")

# 关联规则
t0 = time.time()
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
rules = rules[rules['lift'] > 1]
rules = rules.sort_values('lift', ascending=False)
print(f"关联规则数: {len(rules)}")
print(f"规则生成耗时: {time.time()-t0:.1f}s")

# 筛选后件为平台的规则 vs 其他规则
def has_platform_conseq(csq):
    return any(item.startswith('平台=') for item in csq)

rules_with_platform = rules[rules['consequents'].apply(has_platform_conseq)]
rules_other = rules[~rules['consequents'].apply(has_platform_conseq)]
print(f"  后件为平台: {len(rules_with_platform)} 条")
print(f"  后件非平台: {len(rules_other)} 条")

# 保存规则
def fmt_itemset(s):
    return ', '.join(sorted(s))

rules_save = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
rules_save['antecedents'] = rules_save['antecedents'].apply(fmt_itemset)
rules_save['consequents'] = rules_save['consequents'].apply(fmt_itemset)
rules_save = rules_save.sort_values('lift', ascending=False)
rules_save.to_csv(r'D:\LLY\毕业实训test\association_rules.csv', index=False, encoding='utf-8-sig')
print("-> association_rules.csv")

print("\n=== Top 20 关联规则 (按Lift降序) ===")
for i, (_, r) in enumerate(rules_save.head(20).iterrows()):
    ant = r['antecedents'][:80] + ('...' if len(r['antecedents']) > 80 else '')
    con = r['consequents'][:60] + ('...' if len(r['consequents']) > 60 else '')
    print(f"  {i+1:2d}. [{ant}] => [{con}]")
    print(f"      sup={r['support']:.4f} conf={r['confidence']:.3f} lift={r['lift']:.2f}")
    print()

# ==========================================
# 阶段四：可视化
# ==========================================
print("\n" + "=" * 60)
print("阶段四：综合可视化")
print("=" * 60)

# 图3: 城市×行业热力图
pivot = df.pivot_table(index='city_group', columns='industry',
                        values='avg_salary', aggfunc='count', fill_value=0)
top_cities = df['city_group'].value_counts().head(8).index
top_inds = df['industry'].value_counts().head(8).index
pivot_f = pivot.loc[[c for c in top_cities if c in pivot.index],
                     [c for c in top_inds if c in pivot.columns]]
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pivot_f, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
            linewidths=0.5)
ax.set_title('City x Industry Job Count Heatmap')
ax.set_xlabel('Industry'); ax.set_ylabel('City')
plt.tight_layout()
plt.savefig(r'D:\LLY\毕业实训test\fig03_city_industry_heatmap.png', dpi=150)
plt.close()
print("-> fig03_city_industry_heatmap.png")

# 图4: 岗位层级×行业堆叠柱状图
ct = pd.crosstab(df['industry'], df['cluster_name'])
fig, ax = plt.subplots(figsize=(12, 6))
ct.loc[ct.sum(axis=1).sort_values(ascending=False).head(10).index].plot(
    kind='bar', stacked=True, ax=ax, colormap='Set2')
ax.set_title('Industry x Job Level Distribution')
ax.set_xlabel('Industry'); ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=30)
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig(r'D:\LLY\毕业实训test\fig04_industry_cluster_bar.png', dpi=150)
plt.close()
print("-> fig04_industry_cluster_bar.png")

# 图5: 城市×岗位层级堆叠柱状图
ct2 = pd.crosstab(df['city_group'], df['cluster_name'])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ct2_abs = ct2.loc[ct2.sum(axis=1).sort_values(ascending=False).head(10).index]
ct2_pct = ct2_abs.div(ct2_abs.sum(axis=1), axis=0) * 100
ct2_abs.plot(kind='bar', stacked=True, ax=axes[0], colormap='Set2')
axes[0].set_title('City x Job Level (Absolute)')
axes[0].set_xlabel('City'); axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=30)
ct2_pct.plot(kind='bar', stacked=True, ax=axes[1], colormap='Set2')
axes[1].set_title('City x Job Level (Percentage %)')
axes[1].set_xlabel('City'); axes[1].set_ylabel('Percentage(%)')
axes[1].tick_params(axis='x', rotation=30)
axes[1].legend(fontsize=7, loc='upper right')
plt.tight_layout()
plt.savefig(r'D:\LLY\毕业实训test\fig05_city_cluster_bar.png', dpi=150)
plt.close()
print("-> fig05_city_cluster_bar.png")

# 图6: 薪资分布箱线图（按岗位层级）
fig, ax = plt.subplots(figsize=(10, 5))
sal_sample = df[df['avg_salary'] < 50000].sample(min(8000, len(df)), random_state=42)
sns.boxplot(data=sal_sample, x='cluster_name', y='avg_salary', palette='Set2', ax=ax)
ax.set_title('Salary Distribution by Job Level')
ax.set_xlabel('Job Level')
ax.set_ylabel('Average Salary (CNY)')
ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig(r'D:\LLY\毕业实训test\fig06_salary_boxplot.png', dpi=150)
plt.close()
print("-> fig06_salary_boxplot.png")

# 图7: 关联规则散点图
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(rules['support'], rules['confidence'], c=rules['lift'],
                s=rules['lift'] * 15, cmap='Reds', alpha=0.6, edgecolors='gray', linewidths=0.5)
top8 = rules.nlargest(8, 'lift')
for _, r in top8.iterrows():
    ant = fmt_itemset(r['antecedents'])[:40]
    ax.annotate(ant, (r['support'], r['confidence']),
                fontsize=6, alpha=0.8, xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('Support'); ax.set_ylabel('Confidence')
ax.set_title('Association Rules Scatter (Color/Size = Lift)')
cbar = plt.colorbar(sc)
cbar.set_label('Lift')
plt.tight_layout()
plt.savefig(r'D:\LLY\毕业实训test\fig07_rules_scatter.png', dpi=150)
plt.close()
print("-> fig07_rules_scatter.png")

# 图8: 关联规则网络图
try:
    import networkx as nx
    top_rules_net = rules.nlargest(35, 'lift')
    G = nx.DiGraph()
    for _, r in top_rules_net.iterrows():
        for ant in r['antecedents']:
            short_ant = ant.split('=')[-1] if '=' in ant else ant
            for con in r['consequents']:
                short_con = con.split('=')[-1] if '=' in con else con
                if not G.has_edge(short_ant, short_con):
                    G.add_edge(short_ant, short_con, weight=r['lift'])
    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)
    edges = G.edges()
    weights = [G[u][v]['weight'] * 2 for u, v in edges]
    edge_colors = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue',
                           edgecolors='gray', ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, edge_color=edge_colors,
                           edge_cmap=plt.cm.Reds, alpha=0.6, arrows=True,
                           arrowsize=12, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax)
    ax.set_title('Association Rule Network (Top 35, by Lift)', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(r'D:\LLY\毕业实训test\fig08_rule_network.png', dpi=150)
    plt.close()
    print("-> fig08_rule_network.png")
except Exception as e:
    print(f"网络图跳过: {e}")

# ==========================================
# 阶段五：结论 & 策略
# ==========================================
print("\n" + "=" * 60)
print("阶段五：结论与运营策略")
print("=" * 60)

# 保存处理后的数据
final_cols = ['industry', 'city_group', 'platform_group', 'avg_salary',
              'salary_bin', 'edu_level', 'exp_value', 'cluster_name']
df[final_cols].to_csv(r'D:\LLY\毕业实训test\processed_data.csv', index=False, encoding='utf-8-sig')

# 摘要统计
print(f"\n数据概览:")
print(f"  有效岗位数: {len(df):,}")
print(f"  城市数(Top7+其他): {df['city_group'].nunique()}")
print(f"  行业大类: {df['industry'].nunique()}")
print(f"  岗位层级: {best_k}")
print(f"  关联规则总数: {len(rules)}")

# 各行业薪资统计
ind_salary = df.groupby('industry')['avg_salary'].agg(['mean', 'count']).round(0)
ind_salary = ind_salary[ind_salary['count'] > 100].sort_values('mean', ascending=False)
print(f"\n各行业平均薪资 Top 10:")
print(ind_salary.head(10).to_string())

# 各平台（来源渠道）规则解读
print(f"\n--- 平台关联规则分析 ---")
if len(rules_with_platform) > 0:
    for _, r in rules_with_platform.nlargest(5, 'lift').iterrows():
        ant = fmt_itemset(r['antecedents'])
        con = fmt_itemset(r['consequents'])
        print(f"  [{ant}] => [{con}] (lift={r['lift']:.2f})")
else:
    # 没有平台相关规则时，展示最强城市/行业规则
    print("数据仅包含单一平台(智联招聘)，以下是强关联的城市×行业×薪资规则：")
    for _, r in rules_other.nlargest(10, 'lift').iterrows():
        ant = fmt_itemset(r['antecedents'])[:60]
        con = fmt_itemset(r['consequents'])[:60]
        print(f"  [{ant}] => [{con}] lift={r['lift']:.2f} conf={r['confidence']:.3f}")

print(f"\n所有文件已生成完毕！")
