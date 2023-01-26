import tensorflow as tf
import pandas as pd
import numpy as np

from typing import Tuple, List
from config_allin1out import Config
from sklearn import preprocessing


def window(
    df= pd.DataFrame,
    size= int,
    driving_series= List[str],
    target_series= List[str],
):

    X_list = df[driving_series]
    y_list = df[target_series]
    X_T = []
    y_T = []

    for l in range (len(X_list)):
        X = X_list[l].values
        y = y_list[l].values

        for i in range(len(X) - size + 1):
            X_T.append(X[i : i + size])
            y_T.append(y[i : i + size])


    return np.array(X_T), np.array(y_T)


def window_plus(
    df= pd.DataFrame,
    size= int,
    driving_series= List[str],
    target_series= List[str],
    alpha_series= List[str],
):
    X = df[driving_series].values
    y = df[target_series].values
    alpha = df[alpha_series].values
    X_T = []
    y_T = []
    alpha_T = []
    for i in range(len(X) - size + 1):
        X_T.append(X[i : i + size])
        y_T.append(y[i : i + size])
        alpha_T.append(alpha[i : i + size])

    return np.array(X_T), np.array(y_T), np.array(alpha_T)


def window_series(
    df= pd.DataFrame,
    size= int,
    series= List[str], seq = bool):
    XYal = df[series].values

    XYal_T = []
    for i in range(len(XYal) - size + 1):
        XYal_T.append(XYal[i : i + size])

    return np.array(XYal_T)

 
def get_np_dataset(config= Config, cat_before_window = False, alpha_update= False):  
    dfs = []
    if (alpha_update == True):


        for path in config.data_paths:  
            index = config.data_paths.index(path)
            if index in range(0,int(len(config.data_paths)/3)):
                dfs.append(pd.read_csv(path, sep=config.sep, usecols= config.usecols_driving[index], encoding="utf-8"))
            elif index in range(int((len(config.data_paths))/3), 2*int((len(config.data_paths))/3)):
                index_init = int((len(config.data_paths))/3)
                dfs.append(pd.read_csv(path, sep=config.sep, usecols= config.usecols_target[index-index_init], encoding="utf-8"))
            else:
                index_init = 2* int((len(config.data_paths))/3)
                dfs.append(pd.read_csv(path, sep=config.sep, usecols= config.usecols_alpha[index-index_init], encoding="utf-8"))

    else:


        for path in config.data_paths[0:2*int(((len(config.data_paths))/3))]:  
            index = config.data_paths.index(path)
            if index in range(0,int((len(config.data_paths))/3)):
                dfs.append(pd.read_csv(path, sep=config.sep, usecols= config.usecols_driving[index], encoding="utf-8"))
            else: 
                index_init = int((len(config.data_paths))/3)
                dfs.append(pd.read_csv(path, sep=config.sep, usecols= config.usecols_target[index-index_init], encoding="utf-8"))



    df = None
    X_T = None
    y_T = None
    alpha_T = None
    if cat_before_window:
        df = pd.concat(dfs)
        if (alpha_update == True):
            X_T, y_T, alpha_T = window_plus(
                df, config.T, config.driving_series, config.target_series, config.alpha_series)  
            alpha_T = alpha_T.transpose((0, 2, 1)) 
        else:
            X_T, y_T = window(
                df, config.T, config.driving_series, config.target_series)  

        X_T = X_T.transpose((0, 2, 1)) 
        
    else:
        X_Ts = []
        y_Ts = []
        alpha_Ts = []
        df_nb = 0
        for df in dfs:
            if df_nb in range(0,37): 
                X_T= window_series(df, config.T, config.driving_series[df_nb], seq = True)
                X_T = X_T.transpose((0, 2, 1))
                X_Ts.append(X_T)
            elif df_nb in range(37, 74): 
                y_T= window_series(df, config.T -1 + config.nb_steps_ahead, config.target_series[df_nb-37], seq=True)

                y_Ts.append(y_T)
            else:   
                alpha_T= window_series(df, config.T, config.alpha_series[df_nb-74], seq = True)
                alpha_T = alpha_T.transpose((0, 2, 1))
                alpha_Ts.append(alpha_T)

            df_nb += 1

        X_T = np.vstack(X_Ts)
        y_T = np.vstack(y_Ts)
        if (alpha_update == True):
            alpha_T = np.vstack(alpha_Ts)
            return X_T, y_T, alpha_T
        else:
            return X_T, y_T



def get_datasets(config: Config, cat_before_window: bool = False, shuffled: bool = True, alpha_update: bool = False):  
    """
    Returns X and y of the data passed as config.

    Parameters
    ----------
    config : Config
    cat_before_window : bool
        Whether to concatenate the files before transforming it
        into windows

    Returns
    -------
    train_d : tensorflow.data.Dataset(tuples)
        The tuples are
            X : (config.batch_size, config.n, config.T)
            y : (config.batch_size, config.T, 1)
    val_d
    test_d

    Usage
    -----

    Graph Mode:
    ```
    dataset = get_train_test_dataset(config)
    dataset = dataset.batch(batch_size) # .shuffle() if necessary
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    for _ in range(epochs):
        sess.run(iterator.initializer)
        while True:
            try:
                sess.run(next_element)
            except tf.errors.OutOfRangeError:
                break

        # [Perform end-of-epoch calculations here.]
    ```

    Eager Mode:
    ```
        dataset = get_train_test_dataset(config)
        it = dataset.batch(batch_size)
        for x, y in it:
            print(x, y)
    ```
    """

    if (alpha_update == True):
        X_T, y_T, alpha_T = get_np_dataset(config, cat_before_window, alpha_update)
    else:
        X_T, y_T = get_np_dataset(config, cat_before_window, alpha_update)
    print("get dataset X_T shape=", X_T.shape)
    print("get_datasets y_T shape=", y_T.shape)
    train_size = int(len(X_T) * config.train_ratio)  
    print("train size = ", train_size)
    if (config.train_ratio == 0.0):
        val_size = int(len(X_T) * config.train_ratio)
        test_size = int(len(X_T) * (1-config.train_ratio))
    else:
        val_size = int(((1 - config.train_ratio) / 2) * len(X_T))
        print("val size = ", val_size) 
        test_size = val_size

    if (alpha_update == True):
        dataset = tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(X_T),
                tf.data.Dataset.from_tensor_slices(y_T),
                tf.data.Dataset.from_tensor_slices(alpha_T),
            )
        )
    else:
        dataset = tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(X_T),
                tf.data.Dataset.from_tensor_slices(y_T),
            )
        )


    print("---------------------Finished zipping data in tf dataset structure ----------------------------------------")

   

    if shuffled:
        print("dataset get shuffled")
        dataset = dataset.shuffle(int(len(X_T)), reshuffle_each_iteration=False)

    train_dataset = dataset.take(train_size)

    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size).take(test_size)
    return train_dataset, val_dataset, test_dataset


# Test
if __name__ == "__main__":
    with open("../conf/JackRabbot_x1_longterm.json") as f:
        config = Config.from_json(f.read())

    tr, val, te = get_datasets(config)

    it_train = tr.make_one_shot_iterator()
    print(next(it_train))

    it_val = val.make_one_shot_iterator()
    print(next(it_val))
