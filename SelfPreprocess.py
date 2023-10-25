import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始数据
df = pd.read_csv(r"D:\AutoEncoder\selfdataset\Dataset1.csv")

# 打乱数据
df_shuffled = df.sample(frac=1, random_state=42)

# 划分数据
train_df, test_df = train_test_split(df_shuffled, test_size=0.3, random_state=42)

# 保存到新的文件
train_df.to_csv(r"D:\AutoEncoder\selfdataset\train.csv", index=False)
test_df.to_csv(r"D:\AutoEncoder\selfdataset\test.csv", index=False)