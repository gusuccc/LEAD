import pandas as pd
import json

# 指定列名
column_names = ['uid', 'iid', 'rating', 'timestamp']

# 读取CSV文件并指定列名
df = pd.read_csv('./amazon-CD/ratings_CDs_and_Vinyl.csv', header=None, names=column_names)

# 删除rating中小于等于3的数据
df_filtered = df[df['rating'] > 3]

# 统计每个uid的数据数
uid_counts = df_filtered.groupby('uid').size().reset_index(name='count')

# 找到符合条件的uid
valid_uids = uid_counts[(uid_counts['count'] >= 10) & (uid_counts['count'] <= 12)]['uid']

# 过滤原数据，只保留符合条件的uid
filtered_df = df_filtered[df_filtered['uid'].isin(valid_uids)]

# 将filtered_df按照uid排序
sorted_filtered_df = filtered_df.sort_values(by='uid')

# 如果需要，保存原始id文件
sorted_filtered_df.to_csv('./amazon-CD/ori_interactions.csv', index=False, header=None)

# 重编码 uid 和 iid
uid_mapping = {original_uid: new_uid for new_uid, original_uid in enumerate(sorted_filtered_df['uid'].unique())}
iid_mapping = {original_iid: new_iid for new_iid, original_iid in enumerate(sorted_filtered_df['iid'].unique())}

# 替换 uid 和 iid
sorted_filtered_df['uid'] = sorted_filtered_df['uid'].map(uid_mapping)
sorted_filtered_df['iid'] = sorted_filtered_df['iid'].map(iid_mapping)

# 准备 JSON 映射
mapper_usr = []
mapper_itm = []

for original_uid, new_uid in uid_mapping.items():
    mapper_usr.append({"user": new_uid, "uid": original_uid})

for original_iid, new_iid in iid_mapping.items():
    mapper_itm.append({"iid": new_iid, "asin": original_iid})

# 保存映射到文件
with open('./amazon-CD/amazon_usr.json', 'w') as f:
    for item in mapper_usr:
        f.write(json.dumps(item) + '\n')  # 每行一个 JSON 对象

with open('./amazon-CD/amazon_itm.json', 'w') as f:
    for item in mapper_itm:
        f.write(json.dumps(item) + '\n')  # 每行一个 JSON 对象

# 重置索引
sorted_filtered_df = sorted_filtered_df.reset_index(drop=True)
sorted_filtered_df['rating'] = 1

# 输出重编码后的数据
print(sorted_filtered_df)

# 如果需要，可以保存重编码id文件
sorted_filtered_df.to_csv('./amazon-CD/cod_interactions.csv', index=False, header=None)
