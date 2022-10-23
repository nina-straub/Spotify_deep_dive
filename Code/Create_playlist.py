import utillity_functions
from connection import spotify_connect
import pandas as pd
import spotipy
import numpy as np

config_filepath = "..\99-Private\connect-NiS.csv"
credentials = utillity_functions.get_credentials(config_filepath)

sp = spotify_connect(credentials)

# Add 'playlist-modify-public' to scopes in connect.csv

# Create playlist
def create_playlist(username, playlist_name):
    playlist = sp.user_playlist_create(user = username, name = playlist_name)


# Create list of track ids and chunk list into less than 100 tracks
def create_chunks(dataframe, n):
    uri = dataframe['uri'].tolist()

    id_array = np.array(uri)
    chunk_size = n

    chunked_arrays = np.array_split(id_array, len(uri) / chunk_size)

    chunked_list = [list(array) for array in chunked_arrays]

    return chunked_list


# Add songs to playlist
def add_songs(username, playlist_uri, track_list):
    songs = sp.user_playlist_add_tracks(username, playlist_uri, track_list)



if __name__ == '__main__':
    # read csv to dataframe
    data = pd.read_csv('../Test1/binary_data.csv')  # instead of binary data use streaming history on which classification model was used
    # Filter df for happy songs
    data_happy = data[data['mood'] == 1]

    # create playlist
    create_playlist('ninastraub', 'Happy Songs')

    # Create list of track ids and chunk list into less than 100 tracks
    happy_chunks = create_chunks(data_happy, 50)
    happy_chunks.info()

    # Add tracks to playlist
    add_songs(username = 'ninastraub', playlist_id = '7DkOz9M89o2sGqo7a3uJ9S?si=c2dbd2b0f28143f8', track_list = happy_chunks)
