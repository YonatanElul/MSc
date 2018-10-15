import pickle


"""
"""


def pickle_wfdb_db(db, db_name):
    """

    :param db:
    :param db_name:
    :return:
    """

    with open(db_name, 'wb') as output:
        pickle.dump(db, output, pickle.HIGHEST_PROTOCOL)
