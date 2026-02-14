import pandas as pd
from removing.create_logs import manipulate_log as removing
from changing.create_logs import manipulate_log as changing
from adding.add_events import manipulate_log as adding


file_path = "ground_truth.csv"
df = pd.read_csv(file_path)

df = removing(df, 0.2)
df = df.reset_index(drop=True)
df = changing(df, 0.2)
df = df.reset_index(drop=True)
df = adding(df, 0.2, 0.2)
df.to_csv("final_log.csv", index=False)