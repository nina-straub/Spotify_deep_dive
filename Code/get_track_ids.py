import pandas as pd
import utillity_functions
from connection import spotify_connect
import time

config_filepath = r"..\99-Private\config-AJ.csv"
credentials = utillity_functions.get_credentials(config_filepath)

sp = spotify_connect(credentials)


def track_id_loop(dataframe):
    # Create empty columns 'uri'
    new_col_pos = len(df.columns)
    df.insert(new_col_pos, 'uri', '')

    for row in range(len(dataframe)):
        # Search Query
        try:
            q = str(f"artist: {dataframe.at[row, 'artistName']} track: {dataframe.at[row, 'trackName']}")
            spotify_track_id = sp.search(q, limit=1)['tracks']['items'][0]['id']
            dataframe.loc[row, 'track_id'] = spotify_track_id
        # Intercept if query result is empty
        except IndexError:
            spotify_track_id = "Error"
        # Save to DataFrame
        df.at[row, 'uri'] = spotify_track_id
        # Sleep to stay below Spotify Rate Limit
        time.sleep(0.2)
        print(f"{row + 1}|{len(dataframe)}")

    print(dataframe.head())


# import json in pandas dataframe
df = pd.read_json('../02-Data/FK/StreamingHistory3.json')
print(df.head())
df.info()

# delete items played for less than 20 seconds
new_df = df.drop(df[df['msPlayed'] < 20000].index)

#df_clean = df[df['msPlayed'] < 20000]
new_df.reset_index(drop=True, inplace=True)
print(new_df.head)

#create short dataframe
short_df = new_df[:20]
print(len(short_df))

if __name__ == '__main__':
    track_id_loop(new_df)

    #drop all rows without track id
    new_df = new_df.dropna(axis=0)

    #reset index
    new_df.reset_index(drop=True, inplace=True)

    # Save short_df as csv (or new_df when track_id_loop was used on new_df (= all tracks)
    utillity_functions.save_df(new_df, name='history3_FK_track_ids.csv')

