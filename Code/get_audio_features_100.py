'''import section'''

import pandas as pd
import spotipy
import time
import json
import numpy as np

'''Build connection to spotify API'''

import utillity_functions
from connection import spotify_connect

config_filepath = r"..\99-Private\config-AJ.csv"
credentials = utillity_functions.get_credentials(config_filepath)

sp = spotify_connect(credentials)


'''Read dataframe'''

#history_df = pd.read_csv("../01_Code/indie_covers.csv")
#history_df.info()

#transformation to list (nachfolgend) in Funktion enthalten
#ids = df['track_id'].tolist()
#print(len(ids))
#print(ids)


'''Split list into lists with less than 100 entries"'''
#Look at df.info to choose n accordingly to length of dataframe

def create_chunks(dataframe, n):
    ids = dataframe['track_id'].tolist()

    id_array = np.array(ids)
    chunk_size = n

    chunked_arrays = np.array_split(id_array, len(ids) / chunk_size)

    chunked_list = [list(array) for array in chunked_arrays]

    return chunked_list

#chunked_history = create_chunks(history_df, 50)
#print(chunked_history)


'''Audio features zu allen tracks aus der History'''

def get_audio_features(list):
    df = pd.DataFrame()

    for array in list:
        try:
            features = sp.audio_features(array)
            dfs = [df, pd.DataFrame(features)]
            df = pd.concat(dfs, ignore_index= True)
        except:
            df_= pd.DataFrame()
            for x in array:
                try:
                    features = sp.audio_features(x)
                    dfs_ = [df_, pd.DataFrame(features)]
                    df_ = pd.concat(dfs_, ignore_index=True)
                except:
                    data = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
                    features = pd.dataframe(data)
                    dfs_ = [df_, pd.DataFrame(features)]
                    df_ = pd.concat(dfs_, ignore_index=True)
            df = pd.concat([df, df_], ignore_index=True)
            continue
    return df

#history_features = get_audio_features(chunked_history)
#history_features.info()

#history_features.to_csv("history_data.csv", sep = ',')

'''Ein Dataframe mit allen Infos erstellen'''
#vieleicht hier noch eine geeignetere Funktion

#results = history_df.join(history_features)
#results.info()

#results.to_csv("results.csv", sep = ',')

if __name__ == '__main__':
    #read csv to dataframe
    history_df = pd.read_csv("../03-Output/history1_FK_track_ids.csv")
    history_df.drop(history_df[history_df['artistName'] == 'Die drei ???'].index, inplace = True)
    #history_df = history_df.dropna(axis=0)
    history_df.info()

    #get list of ids out of dataframe and split into chunks
    chunked_history = create_chunks(history_df, 50)
    #print(chunked_history)

    #Get Audio features of all tracks (more than 100)
    history_features = get_audio_features(chunked_history)
    history_features.info()

    #history_features.to_csv("history_data.csv", sep=',')

    #rename track_id to id
    history_df.rename(columns={'track_id': 'id'}, inplace=True)
    history_df['id'] = history_df['id'].astype(str)
    history_features['id'] = history_features['id'].astype(str)

    #history_df.to_csv('test_history.csv')
    #history_features.to_csv('test_history_features.csv')

    history_df.reset_index(drop=True, inplace=True)
    history_features.reset_index(drop=True, inplace=True)
    results = pd.concat([history_df, history_features], axis=1)

    #results = history_df.set_index('id').join(history_features.set_index('id'))

    print(f"history_df: {history_df.shape}")
    print(f"history_features: {history_features.shape}")
    print(f"results: {results.shape}")


    #Put together dataframe  with ids from beginning with audiofeature dataframe
    #results = history_df.join(history_features, on='id')
    results.info()

    results.to_csv("new_history_1_FK.csv", sep = ',')

