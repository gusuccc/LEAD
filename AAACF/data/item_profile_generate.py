import json
import gzip
import pandas as pd
from tqdm import tqdm  # 导入 tqdm
# amazon数据集官方读法，读到pandas的df中
# 读取mapper就好
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
# df.head()
meta = df[['asin', 'title', 'description']]
# 生成item_prompt
# 打开原始 JSON 文件
with open('./amazon-CD/amazon_itm.json', 'r') as file:
    # 读取所有行以获取总线数
    lines = file.readlines()

# 打开输出文件
with open('./amazon-CD/itm_prompts.json', 'w') as output_file:
    for line in tqdm(lines, desc="Processing items"):
        item = json.loads(line)
        iid = item['iid']
        asin = item['asin']

        # 获取标题和描述，并处理 NaN 值
        title = meta[meta['asin'] == asin]['title'].values[0]
        description = meta[meta['asin'] == asin]['description'].values[0]

        # 替换 NaN 值
        if pd.isna(description):
            description = "None"  # 或者使用空字符串 ""
        # 替换 NaN 值
        if pd.isna(title):
            title = "None"  # 或者使用空字符串 ""

        # 合并标题和描述
        combined_info = f"BASIC INFORMATION: \n{{\"title\": \"{title}\", \"description\": \"{description}\"}}"

        # 构建 prompt 字典
        prompt_dict = {
            "prompt": combined_info
        }

        # 写入输出文件
        output_file.write(json.dumps(prompt_dict) + "\n") # 行id = iid，从0开始

print("Prompts have been written to prompts.json")