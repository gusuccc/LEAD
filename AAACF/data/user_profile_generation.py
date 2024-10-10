import pandas as pd
import gzip
import json
from tqdm import tqdm  # 导入 tqdm

# 读取原始id数据集
column_names = ['uid', 'iid', 'rating', 'timestamp']
df = pd.read_csv('./amazon-CD/ori_interactions.csv',header=None,names=column_names)
df =df[['uid', 'iid']]
# 初始化一个字典来存储用户和他们的iid列表
user_interactions = {}

# 遍历每一行，统计每个用户交互的iid
for index, row in df.iterrows():
    uid = row['uid']
    iid = row['iid']

    # 如果这个用户还没有记录，则初始化一个空列表
    if uid not in user_interactions:
        user_interactions[uid] = []

    # 将iid添加到用户的交互列表中
    user_interactions[uid].append(iid)

# 如果需要将结果转换为列表形式
# 这将按顺序排列用户交互的iid，假设用户的 uid 是有序且连续的
user_list = list(user_interactions.values())
# amazon数据集官方读法，读到pandas的df中
path = './amazon-CD/meta_CDs_and_Vinyl.json.gz'

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF(path)
meta = df[['asin', 'title', 'description']]
df = getDF('./amazon-CD/reviews_CDs_and_Vinyl_5.json.gz')
review = df[['reviewerID', 'asin', 'reviewText']]

# 打开原始 JSON 文件
with open('./amazon-CD/amazon_usr.json', 'r') as file:
    # 读取所有行以获取总线数
    lines = file.readlines()

# 打开输出文件

with open('./amazon-CD/u_prompts.json', 'w') as output_file:
    for line in tqdm(lines, desc="Processing items"):
        item = json.loads(line)
        user = item['user']  # 重编码
        uid = item['uid']  # 原编码
        # print(uid)
        # 获取asin和title,description和reviewText，并处理 NaN 值
        user_reviews = review[review['reviewerID'] == uid]
        # print(user_reviews)
        # 初始化列表以存储所有的 title 和 description
        titles_descriptions_reviews = []

        if user_reviews.empty:
            # 如果没有匹配的 review，只是用交互数据
            for count, asin in enumerate(user_list[user]):
                if count >= 5:
                    break
                title = meta[meta['asin'] == asin]['title'].values[0]
                description = meta[meta['asin'] == asin]['description'].values[0]
                reviewText = "None"
                if pd.isna(description):
                    description = "None"
                if pd.isna(title):
                    title = "None"
                # 将 title 和 description 添加到列表中
                titles_descriptions_reviews.append({"title": title, "description": description, "review": reviewText})
            # 构建信息字符串
            combined_info = "PURCHASED CDs and Vinyl:\n" + json.dumps(titles_descriptions_reviews)

        else:
            # 遍历不超过5个的 user_reviews
            for count, (_, row) in enumerate(user_reviews.iterrows()):
                if count >= 5:
                    break
                asin = row['asin']
                title = meta[meta['asin'] == asin]['title'].values[0]
                description = meta[meta['asin'] == asin]['description'].values[0]
                reviewText = row['reviewText']

                # 替换 NaN 值
                if pd.isna(title):
                    title = "None"
                if pd.isna(description):
                    description = "None"
                if pd.isna(reviewText):
                    reviewText = "None"

                # 将 title, description 和 review 添加到列表中
                titles_descriptions_reviews.append({
                    "title": title,
                    "description": description,
                    "review": reviewText
                })

            # 构建信息字符串
            combined_info = "PURCHASED CDs and Vinyl:\n" + json.dumps(titles_descriptions_reviews)

        # 构建 prompt 字典
        prompt_dict = {
            "prompt": combined_info
        }

        # 写入输出文件
        output_file.write(json.dumps(prompt_dict) + "\n")

print("Prompts have been written to prompts.json")