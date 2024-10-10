import pandas as pd

data_path = '/data7_8T/ycx/tmp/Recbole_STRec/RecBole-GNN/data/Amazon_Books/ratings_Books.csv'
df = pd.read_csv(data_path,header=None)

# 打印最后几行，查看是否有空行
print(df.tail())

# 查看所有用户ID
user_ids = df[0].dropna().astype(int)
print(f"所有用户ID: {user_ids}")
unique_user_ids = user_ids.unique()
print(f"总用户数: {len(unique_user_ids)}")
print(f"用户ID范围: {unique_user_ids.min()} - {unique_user_ids.max()}")

# 检查是否有重复的用户ID
if len(unique_user_ids) != len(user_ids):
    print("存在重复的用户ID")

# 检查用户ID是否连续
expected_user_ids = set(range(unique_user_ids.min(), unique_user_ids.max() + 1))
actual_user_ids = set(unique_user_ids)

missing_user_ids = expected_user_ids - actual_user_ids
if missing_user_ids:
    print(f"缺失的用户ID: {missing_user_ids}")
else:
    print("用户ID是连续的")
