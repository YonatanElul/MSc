import tensorflow as tf
import os


def _int64_feature(value):
    """
    Description:
    This function convert the supplied value into an int64 feature for tensorflow

    :param value: Data object to be converted
    :return: TF int64 feature
    """

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """
    Description:
    This function convert the supplied value into a float feature for tensorflow

    :param value: Data object to be converted
    :return: TF float feature
    """

    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """
    Description:
    This function convert the supplied value into a bytes feature for tensorflow

    :param value: Data object to be converted
    :return: TF bytes feature
    """

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def db_tfrecord_convert(database, file_name: str, file_save_path: str):
    """
    Description:
    This function convert a database object into a tensorflow tf_recotd object

    :param database: The database to convert
    :param file_name:
    :param file_save_path: The path in which to save tf_record file
    :return: None - The file is written in the specified path
    """

    # Setup
    labels = database.labels
    num_examples = database.shape[0]
    signal_length = database.shape[1]
    num_of_features = database.shape[3]
    filename = os.path.join(file_save_path, file_name + '.tfrecords')

    print('Writing', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
            raw_example = database[index].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'signal_length': _int64_feature(signal_length),
                        'num_of_features': _int64_feature(num_of_features),
                        'label': _int64_feature(int(labels[index])),
                        'raw_signal': _bytes_feature(raw_example)
                    }))

            writer.write(example.SerializeToString())
