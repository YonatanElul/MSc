from ML.Utilities.DBHandler import DBHandler
from ML.Utilities.PickleDB import pickle_wfdb_db

import os


def create_db_pickles(raw_data_dir: str, save_to_dir: str, db_name: str, db_type: str):
    """
    Description:
    This function pickles a given database from raw files in the wfdb format. The pickled file can be later used in
    the various 'ML' and 'DeterministicAnalytics' packages.

    :param raw_data_dir: str - Path to the directory containing all of the raw files to be pickle
    :param save_to_dir: str - The path to the directory in which to save the pickle
    :param db_name: str - The name of database to be saved
    :param db_type: str - The type of the database, to be used in the DBHandler
    :return:
    """

    print('\n' + 'Pickeling ' + db_name + '\n')
    db = DBHandler(raw_data_dir, db_type)
    pickle_wfdb_db(db, os.path.join(save_to_dir, db_name + '.pkl'))
    del db

