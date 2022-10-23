'''import section'''

import pandas as pd
import numpy as np
from joblib import dump, load

'''Build connection to spotify API'''

import utillity_functions
from connection import spotify_connect

config_filepath = "../99-Private/Caro_Credentials.cfg"
credentials = utillity_functions.get_credentials(config_filepath)

sp = spotify_connect(credentials)

'''Load Random Forest'''

randomforest = load('Mood_RandomForest.joblib')

'''Read predict dataframes'''

df1 = pd.read_csv("../01_Code/history_0_CS.csv")
df2 = pd.read_csv("../01_Code/history_1_CS.csv")
df3 = pd.read_csv("../01_Code/history_2_CS.csv")
dfs = [df1, df2, df3]
history_CS = pd.concat(dfs)
history_CS.drop_duplicates(subset=['id'], keep= 'first', inplace=True)
history_CSx = history_CS.drop(columns = ['Unnamed: 0','id','type','uri','track_href','analysis_url','time_signature'], axis = 1)
print('Caro')
history_CS.info()

df1 = pd.read_csv("../01_Code/history_0_AJ.csv")
df2 = pd.read_csv("../01_Code/history_1_AJ.csv")
dfs = [df1, df2]
history_AJ = pd.concat(dfs)
history_AJ.drop_duplicates(subset=['id'], keep= 'first', inplace=True)
history_AJx = history_AJ.drop(columns = ['Unnamed: 0','id','type','uri','track_href','analysis_url','time_signature'], axis = 1)
print('Alex')
history_AJ.info()

df1 = pd.read_csv("../01_Code/history_0_DA.csv")
df2 = pd.read_csv("../01_Code/history_1_DA.csv")
df3 = pd.read_csv("../01_Code/history_2_DA.csv")
dfs = [df1, df2, df3]
history_DA = pd.concat(dfs)
history_DA.drop_duplicates(subset=['id'], keep= 'first', inplace=True)
history_DAx = history_DA.drop(columns = ['Unnamed: 0','id','type','uri','track_href','analysis_url','time_signature'], axis = 1)
print('Daniel')
#history_DA.info()
history_DAx.info()

df1 = pd.read_csv("../01_Code/history_0_FK.csv")
df2 = pd.read_csv("../01_Code/history_1_FK.csv")
df3 = pd.read_csv("../01_Code/history_2_FK.csv")
df4 = pd.read_csv("../01_Code/history_3_FK.csv")
dfs = [df1, df2, df3, df4]
history_FK = pd.concat(dfs)
history_FK.drop_duplicates(subset=['id'], keep= 'first', inplace=True)

history_FK.drop(3702, axis = 0, inplace = True)
#history_FK.reset_index(drop = True)

history_FKx = history_FK.drop(columns = ['Unnamed: 0','id','type','uri','track_href','analysis_url','time_signature'], axis = 1)
history_FKx.pop('0')

print('Summary of missing values:')
print('------------------------------------')
print(history_FKx.isnull().sum())

print('Find missing values:')
print('------------------------------------')
df_missing = history_FKx[history_FKx.isna().any(axis=1)]
print(df_missing)

history_FKx = history_FKx.dropna()
history_FKx.reset_index()
history_FKx.to_csv("why_tho_Felix.csv", sep=',')


#print(history_FK.isnull().sum())
#df_missing = history_FK[history_FK.isna().any(axis=1)]
#print(df_missing)
#history_FK = history_FK.dropna()
#history_FK.drop(3702, axis = 0)
#history_FK.reset_index(drop = True)

print('Felix')
history_FK.info()
history_FKx.info()


df1 = pd.read_csv("../01_Code/history_0_ME.csv")
df2 = pd.read_csv("../01_Code/history_1_ME.csv")
dfs = [df1, df2]
history_ME = pd.concat(dfs)
history_ME.drop_duplicates(subset=['id'], keep= 'first', inplace=True)
history_MEx = history_ME.drop(columns = ['Unnamed: 0','id','type','uri','track_href','analysis_url','time_signature'], axis = 1)
print('Magnus')
history_ME.info()

df1 = pd.read_csv("../01_Code/history_0_NiS.csv")
df2 = pd.read_csv("../01_Code/history_1_NiS.csv")
dfs = [df1, df2]
history_NiS = pd.concat(dfs)
history_NiS.drop_duplicates(subset=['id'], keep= 'first', inplace=True)
history_NiSx = history_NiS.drop(columns = ['Unnamed: 0','id','type','uri','track_href','analysis_url','time_signature'], axis = 1)
print('Nina')
history_NiS.info()

'''Predict Streaming History'''

CS_history_pred = randomforest.predict(history_CSx)
history_CS['mood'] = CS_history_pred.tolist()



CS_history_pred_proba = randomforest.predict_proba(history_CSx)
pred_proba_list_CS = np.hsplit(CS_history_pred_proba, 4)

history_CS['prd_calm'] = pred_proba_list_CS[0]
history_CS['prd_energetic'] = pred_proba_list_CS[1]
history_CS['prd_happy'] = pred_proba_list_CS[2]
history_CS['prd_sad'] = pred_proba_list_CS[3]


history_CS.to_csv("labeled_history_CS.csv", sep=',')

AJ_history_pred = randomforest.predict(history_AJx)
history_AJ['mood'] = AJ_history_pred.tolist()

AJ_history_pred_proba = randomforest.predict_proba(history_AJx)
pred_proba_list_AJ = np.hsplit(AJ_history_pred_proba, 4)

history_AJ['prd_calm'] = pred_proba_list_AJ[0]
history_AJ['prd_energetic'] = pred_proba_list_AJ[1]
history_AJ['prd_happy'] = pred_proba_list_AJ[2]
history_AJ['prd_sad'] = pred_proba_list_AJ[3]

history_AJ.to_csv("labeled_history_AJ.csv", sep=',')


DA_history_pred = randomforest.predict(history_DAx)
history_DA['mood'] = DA_history_pred.tolist()

DA_history_pred_proba = randomforest.predict_proba(history_DAx)
pred_proba_list_DA = np.hsplit(DA_history_pred_proba, 4)

history_DA['prd_calm'] = pred_proba_list_DA[0]
history_DA['prd_energetic'] = pred_proba_list_DA[1]
history_DA['prd_happy'] = pred_proba_list_DA[2]
history_DA['prd_sad'] = pred_proba_list_DA[3]

history_DA.to_csv("labeled_history_DA.csv", sep=',')

FK_history_pred = randomforest.predict(history_FKx)
history_FK['mood'] = FK_history_pred.tolist()

FK_history_pred_proba = randomforest.predict_proba(history_FKx)
pred_proba_list_FK = np.hsplit(FK_history_pred_proba, 4)

history_FK['prd_calm'] = pred_proba_list_FK[0]
history_FK['prd_energetic'] = pred_proba_list_FK[1]
history_FK['prd_happy'] = pred_proba_list_FK[2]
history_FK['prd_sad'] = pred_proba_list_FK[3]

history_FK.to_csv("labeled_history_FK.csv", sep=',')

ME_history_pred = randomforest.predict(history_MEx)
history_ME['mood'] = ME_history_pred.tolist()

ME_history_pred_proba = randomforest.predict_proba(history_MEx)
pred_proba_list_ME = np.hsplit(ME_history_pred_proba, 4)

history_ME['prd_calm'] = pred_proba_list_ME[0]
history_ME['prd_energetic'] = pred_proba_list_ME[1]
history_ME['prd_happy'] = pred_proba_list_ME[2]
history_ME['prd_sad'] = pred_proba_list_ME[3]

history_ME.to_csv("labeled_history_ME.csv", sep=',')


NiS_history_pred = randomforest.predict(history_NiSx)
history_NiS['mood'] = NiS_history_pred.tolist()

NiS_history_pred_proba = randomforest.predict_proba(history_NiSx)
pred_proba_list_NiS = np.hsplit(NiS_history_pred_proba, 4)

history_NiS['prd_calm'] = pred_proba_list_NiS[0]
history_NiS['prd_energetic'] = pred_proba_list_NiS[1]
history_NiS['prd_happy'] = pred_proba_list_NiS[2]
history_NiS['prd_sad'] = pred_proba_list_NiS[3]

history_NiS.to_csv("labeled_history_NiS.csv", sep=',')


'''Select most happy songs'''
happy_songs_CS = history_CS[history_CS['mood'] == 'happy']
most_happy_songs_CS = happy_songs_CS.sort_values(by=['prd_happy'], ascending = False)
most_happy_songs_CS.to_csv('CS_most_happy.csv', sep = ',')
list_CS = most_happy_songs_CS['uri'].to_list()
list_CS= list_CS[:75]
print(list_CS)
#print(len(list_CS))

happy_songs_AJ = history_AJ[history_AJ['mood'] == 'happy']
most_happy_songs_AJ = happy_songs_AJ.sort_values(by=['prd_happy'], ascending = False)
list_AJ = most_happy_songs_AJ['uri'].to_list()
list_AJ= list_AJ[:75]

happy_songs_DA = history_DA[history_DA['mood'] == 'happy']
most_happy_songs_DA = happy_songs_DA.sort_values(by=['prd_happy'], ascending = False)
list_DA = most_happy_songs_DA['uri'].to_list()
list_DA= list_DA[:75]

happy_songs_FK = history_FK[history_FK['mood'] == 'happy']
most_happy_songs_FK = happy_songs_FK.sort_values(by=['prd_happy'], ascending = False)
list_FK = most_happy_songs_FK['uri'].to_list()
list_FK= list_FK[:75]


happy_songs_ME = history_ME[history_ME['mood'] == 'happy']
most_happy_songs_ME = happy_songs_ME.sort_values(by=['prd_happy'], ascending = False)
list_ME = most_happy_songs_ME['uri'].to_list()
list_ME= list_ME[:75]

happy_songs_NiS = history_NiS[history_NiS['mood'] == 'happy']
most_happy_songs_NiS = happy_songs_NiS.sort_values(by=['prd_happy'], ascending = False)
list_NiS = most_happy_songs_NiS['uri'].to_list()
list_NiS= list_NiS[:75]


#lists = [list_CS, list_AJ, list_DA, list_ME, list_NiS] # list_FK
most_happy_songs_all = list_CS + list_AJ + list_DA + list_ME + list_NiS + list_FK
most_happy_songs_all = list(dict.fromkeys(most_happy_songs_all))
print(len(most_happy_songs_all))


most_happy_songs_all1 = most_happy_songs_all[:100]
most_happy_songs_all2 = most_happy_songs_all[101:200]
most_happy_songs_all3 = most_happy_songs_all[201:300]
most_happy_songs_all4 = most_happy_songs_all[301:]



#happy_songs_CS = history_CS[history_CS['mood'] == 'happy']
#list = happy_songs_CS['uri'].to_list()
#list = list[160:260]
#print(len(list))



'''Create Playlist'''
'''
def create_playlist(self, userID=None, name='testPlaylist', trackIdList=None):
    if userID == None or trackIdList == None:
        print('userID and trackIds cannot be None')
        return

    user_id = sp.me()['id']
    print(user_id)

    output = sp.user_playlist_create(user_id, name, public = True)
    playlistId = output['id']
    sp.user_playlist_add_tracks(user_id, playlistId, trackIdList[:50])


TrackList = most_happy_songs_all

create_playlist('carolilalelolu', 'Most Happy Songs', TrackList)
'''
'''
#def create_playlist(username,playlist_name):
#    playlist = sp.user_playlist_create(user=username, name=playlist_name)

#create_playlist('carolilalelolu', 'Most Happy Songs')


#https://open.spotify.com/playlist/1Uuje9z7IJVn83Spk2XmtY?si=392b7a8c82344d8d
'''
#add tracks to playlist
'''
def add_songs(username, playlist_id, tracklist):
    songs = sp.user_playlist_add_tracks(username, playlist_id, tracklist)


add_songs('carolilalelolu','1Uuje9z7IJVn83Spk2XmtY', most_happy_songs_all1)
add_songs('carolilalelolu','1Uuje9z7IJVn83Spk2XmtY', most_happy_songs_all2)
add_songs('carolilalelolu','1Uuje9z7IJVn83Spk2XmtY', most_happy_songs_all3)
add_songs('carolilalelolu','1Uuje9z7IJVn83Spk2XmtY', most_happy_songs_all4)

