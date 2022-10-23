import csv


def print_dict(dictionary: dict) -> None:
    """
    Quick function to print a dictionary
    :param dictionary: dictionary to print
    :return: None
    """
    for key, value in dictionary.items():
        print("{}: {}".format(key, value))


def save_df(df: object, name='unnamed.csv') -> None:
    """
    Quick function to save a DataFrame as csv in the output folder
    :param df: Dataframe to save
    :param name: name of the csv file
    :return: None
    """
    path = '../03-Output/' + name
    df.to_csv(path, sep=',', encoding='utf-8', index=False)


def get_credentials(config_filepath: str) -> dict:
    """
    Read the credentials from your config.cfg file and stores it into a dictionary
    :param config_filepath: config file with
    :return: dictionary of all credentials
    """
    with open(config_filepath, mode='r') as infile:
        reader = csv.reader(infile)
        credentials = {rows[0]: rows[1] for rows in reader}
    return credentials
