"""
Sammlung aller Funktionen, die zur Visualisierung genutzt werden
"""
import pandas as pd
import plotly.express as px

#read csv file
df = pd.read_csv("../03-Output/AJ_merged.csv")

df.info()

#get means of audio features
means_float = df[["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]].mean()
print(means_float)

# transform float object into list
means_list = list(means_float)
print(means_list)

# create df with means and features
df_plot = pd.DataFrame(dict(
    means = means_list,
    feature = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']))

# radar plot
fig = px.line_polar(df_plot, r='means', range_r=[0,1], theta='feature', line_close=True, title = "Alexandra's mood profile")
fig.show()

#view radar plot
#fig.show()

#save radar plot as image
import os

if not os.path.exists("images"):
    os.mkdir("images")

fig.write_image("images/fig1.png")


