import pandas as pd
import os


def mergeJson(path):
    # returns the merged StreamingHistory files of one path
    # for path enter the directory of your desired data (e.g. "../02-Data/ME")
    keyword = 'Streaming'
    dfs = []
    os.chdir(path)
    for file in os.listdir():
        if file.endswith("json") and keyword in file:
            data = pd.read_json(file)
            dfs.append(data)
    df = pd.concat(dfs, ignore_index=False)

    df_without_skipped = df.drop(df[df['msPlayed'] < 20000].index)

    os.chdir("../")

    return df_without_skipped


