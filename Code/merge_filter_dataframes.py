import pandas as pd
import utillity_functions

#read csv
df1 = pd.read_csv("../03-Output/new_history_0_NiS.csv")
df2 = pd.read_csv("../03-Output/new_history_1_NiS.csv")

df = pd.concat([df1, df2])

# delete certain artists (Podcasts etc.)

# df_clean = df.drop(df[df['artistName'] == 'Hans Joachim Markowitsch'].index)
# df_clean = df_clean.drop(df[df['artistName'] == 'Stephenie Meyer'].index)
# df_clean = df_clean.drop(df[df['artistName'] == 'Mai Thi Nguyen-Kim'].index)

df_clean = df[~df['artistName'].isin(['Hans Joachim Markowitsch', 'Stephenie Meyer', 'Mai Thi Nguyen-Kim'])]

#reset index
df_clean.reset_index(drop=True, inplace=True)

# Save short_df as csv (or new_df when track_id_loop was used on new_df (= all tracks)
utillity_functions.save_df(df_clean, name='NiS_merged.csv')
