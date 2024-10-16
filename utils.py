import numpy as np


def shuffle_dataset(x_train, y_train, x_test, y_test, split_ratio):
    train_ds = np.column_stack((x_train, y_train))
    test_ds = np.column_stack((x_test, y_test))

    combined_data = np.concatenate((train_ds, test_ds), axis=0)
    indices = np.arange(combined_data.shape[0])
    np.random.shuffle(indices)
    shuffled_data = combined_data[indices]

    split_point = int(split_ratio * shuffled_data.shape[0])
    new_train_data = shuffled_data[:split_point]
    new_test_data = shuffled_data[split_point:]
    return new_train_data[:, :-1], new_train_data[:, -1], new_test_data[:, :-1], new_test_data[:, -1]
