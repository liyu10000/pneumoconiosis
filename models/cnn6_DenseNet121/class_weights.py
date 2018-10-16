import numpy as np
import pandas as pd


def get_class_entry_counts(df, classes_key):
    """ get total and class-wise entry counts of dataset
    :param df: pandas dataframe
    :param classes_key: the name of classes, in a list
    :return: total_count, class_counts
    """
    total_count = df.shape[0]
    labels = df[classes_key].as_matrix()
    positive_counts = np.sum(labels, axis=0)
    class_counts = dict(zip(classes_key, positive_counts))
    return total_count, class_counts


def get_class_weights(df, classes_key, multiply=1):
    """ calculate class weights used in training
    :param df: pandas dataframe
    :param classes_key: the name of classes, in a list
    :return: class_weights
    """
    def get_single_class_weight(total_count, class_count):
        denominator = (total_count - class_count) * multiply + class_count
        return {
            0: class_count / denominator,
            1: (denominator - class_count) / denominator
        }

    total_count, class_counts = get_class_entry_counts(df, classes_key)
    class_weights = [get_single_class_weight(total_count, class_counts[class_i]) for class_i in classes_key]
    return class_weights