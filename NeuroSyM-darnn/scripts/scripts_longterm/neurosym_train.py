#!/usr/bin/env python

# email: sariahmghames@gmail.com ========================================================================================================================

import sys
import math, time
import os

import tensorflow as tf
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

tf.compat.v1.enable_eager_execution()
print(tf.executing_eagerly())

import shutil 
import time

from pathlib import Path
import numpy as np
from config_allin1out import Config

import neurosym_model as _model
from data_loader_allin1out import get_datasets
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from qtc import qtcc22 
from sklearn import preprocessing


print("Using TensorFlow version %s" % tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))




flags = tf.compat.v1.app.flags 
FLAGS = flags.FLAGS


flags.DEFINE_string('config1', 'conf/JackRabbot.json', 'Path to json file with the configuration to be run') 
flags.DEFINE_string('config2', 'conf/JackRabbot.json', 'Path to json file with the configuration to be run')

run_validation = True
run_test = False 
run_train = True
alpha_update =  True
inp_att_enabled =  True
temporal_att_enabled =  True

qbits = 4
var = []

epochs_RMSE_val = []
epochs_loss_val = []
epochs_RMSE_train = []
epochs_loss_train = []


epochs_train_y_true = []
epochs_train_y_pred = []
epochs_val_y_true = []
epochs_val_y_pred = []
epochs_train_x = []
epochs_val_x = []





def copy_checkpoint(source, target):
    for ext in (".index", ".data-00000-of-00001"):
        shutil.copyfile(source.with_suffix(ext), target.with_suffix(ext))


def make_summary(value_dict):
    return tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])





def plot_rmse_per_epoch(session, model, next_element, i, log_path: Path, stage: str):
    all_true = []
    all_predicted = []
    label_error = []
    loop = 0
    batch_elements = 0
    config1 = Config.from_file(FLAGS.config1)

    while True:
        try:
            if (alpha_update == True):
                x, y, alpha = session.run(next_element)
                loop += 1
                predictions = session.run(model.predictions, {model.driving_series: x, model.past_history: y, model.alphaAtt: alpha})
            else:
                x, y = session.run(next_element)
                loop += 1 
                predictions = session.run(model.predictions, {model.driving_series: x, model.past_history: y})
            for batch_el in range(predictions.shape[0]):
                error = np.subtract(y[batch_el, -1, :], predictions[batch_el,:])
                predicted_error = np.reshape(error, [-1]).tolist()
                label_error.append(predicted_error)
        except tf.errors.OutOfRangeError:
            break


    # cumulative error over time for each pair of peds
    mean_pred_errors = [np.sqrt(np.mean(np.square(i),axis=0)) for i in zip(*label_error)]
    var_per_epoch = np.var(mean_pred_errors)
    var.append(var_per_epoch)

    x = np.linspace(0, config1.labels, num=config1.labels, endpoint=False)

    plt.figure()
    plt.plot(x, mean_pred_errors, label="rmse")

    plt.legend(loc='upper left')

    plt.title(f"{stage} data")

    plt.ylabel("predicted rmse")
    plt.xlabel("labels")
    plt.savefig(log_path / f"plot{i}-{stage}.png", dpi=300)
    plt.close()




def main(argv): 

    config1 = Config.from_file(FLAGS.config1) 
    config2 = Config.from_file(FLAGS.config2)

    np.random.seed(config1.seed)

    with tf.Graph().as_default():

        # set seeds for reproducibility
        tf.random.set_seed(config1.seed)   # TF v2.0+

        with tf.compat.v1.Session() as session:


            train_set_x, val_set_x, test_set_x = get_datasets(config1, shuffled=True, alpha_update=alpha_update)
            train_set_y, val_set_y, test_set_y = get_datasets(config2, shuffled=True, alpha_update=alpha_update)

            train_set_x = train_set_x.batch(config1.batch_size, drop_remainder=True) # returns them in batches ?
            val_set_x = val_set_x.batch(config1.batch_size, drop_remainder=True)
            test_set_x = test_set_x.batch(config1.batch_size, drop_remainder=True)

            train_set_y = train_set_y.batch(config2.batch_size, drop_remainder=True)
            val_set_y = val_set_y.batch(config2.batch_size, drop_remainder=True)
            test_set_y = test_set_y.batch(config2.batch_size, drop_remainder=True)


            model = _model.TimeAttnModel(config=config1, alpha_update=alpha_update, inp_att=inp_att_enabled, temp_att=temporal_att_enabled)
            print("====================================== MODEL INIT ======================================")
            report_frequency = config1.report_frequency
            saver = tf.compat.v1.train.Saver(max_to_keep=1)   ########################### Saver adds ops to save and restore variables to and from checkpoints,  max_to_keep indicates the maximum number of recent checkpoint files to keep.
            log_path = config1.log_path # path to log dir "log", maybe if log_path is a cached_property we dont add ()
            writer = tf.compat.v1.summary.FileWriter(log_path, flush_secs=20)

            best_RMSE = float("inf")
            best_MAE = float("inf")
            best_MAPE = float("inf")

            accumulated_RMSE = 0.0
            epoch_acc_train = 0.0
            initial_time = time.time()

            session.run(tf.compat.v1.global_variables_initializer())
            tf_global_step = 0

            train_iterator_x = tf.compat.v1.data.make_initializable_iterator(train_set_x) 
            val_iterator_x = tf.compat.v1.data.make_initializable_iterator(val_set_x)
            test_iterator_x = tf.compat.v1.data.make_initializable_iterator(test_set_x)

            train_iterator_y = tf.compat.v1.data.make_initializable_iterator(train_set_y) 
            val_iterator_y = tf.compat.v1.data.make_initializable_iterator(val_set_y)
            test_iterator_y = tf.compat.v1.data.make_initializable_iterator(test_set_y)

            train_next_element_x = train_iterator_x.get_next()   
            val_next_element_x = val_iterator_x.get_next()
            test_next_element_x = test_iterator_x.get_next()

            train_next_element_y = train_iterator_y.get_next()  
            val_next_element_y = val_iterator_y.get_next()
            test_next_element_y = test_iterator_y.get_next()

            # Restore from last evaluated epoch      
            ckpt = tf.compat.v1.train.get_checkpoint_state(log_path)
            if ckpt and ckpt.model_checkpoint_path:
                print("Restoring from: {}".format(ckpt.model_checkpoint_path)) 
                saver.restore(session, ckpt.model_checkpoint_path)   
                init_global_step = session.run(model.global_step) 
            else:
                init_global_step = 0 


            if (run_train == True):

                total_parameters = 0
                for variable in tf.compat.v1.trainable_variables():
                    shape = variable.get_shape()
                    variable_parameters = 1
                    for dim in shape:
                        variable_parameters *= dim 
                    total_parameters += variable_parameters
                print("total nb of params =", total_parameters)

                for i in range(config1.num_epochs):

                    session.run(train_iterator_x.initializer) 
                    session.run(train_iterator_y.initializer)
                    print(f"====================================== EPOCH {i} ======================================")

                    epoch_train_y_true = np.zeros((config1.nb_steps_ahead, 2*config1.labels))
                    epoch_train_y_pred = np.zeros((config1.nb_steps_ahead, 2*config1.labels))
                    epoch_train_x = np.zeros((2*config1.n, config1.T))
                    epoch_batches = 0.0

                    while True:

                        try:

                            if (alpha_update == False):

                                xX, yX= session.run(train_next_element_x) # X coordinate, x is input , y is output
                                xY, yY = session.run(train_next_element_y) # Y coordinate, x is input, y is output


                                X = np.concatenate((xX, xY), axis = 1) # X here is input to net
                                Y = np.concatenate((yX, yY), axis = 1) # Y here is outout to net

                                Y = np.reshape(Y, [config1.batch_size, config1.T-1+config1.nb_steps_ahead, 2*config1.labels])
                                #print("--------------------------------- X shape--------------------------------=", X.shape)

                                train_RMSE, train_pred, tf_global_step, _, _ = session.run([model.RMSE, model.predictions, model.global_step, model.train_op_en, model.train_op_dec], feed_dict= {model.driving_series: X, model.past_history: Y}) 

                            else:

                                xX, yX, Alpha1 = session.run(train_next_element_x) # X coordinate, x is input , y is output
                                xY, yY, Alpha2 = session.run(train_next_element_y) # Y coordinate, x is input, y is output

                                X = np.concatenate((xX, xY), axis = 1) # X here is input to net
                                Y = np.concatenate((yX, yY), axis = 1) # Y here is outout to net
                                alpha = np.concatenate((Alpha1, Alpha2), axis = 1) 
                                Y = np.reshape(Y, [config1.batch_size, config1.T-1+config1.nb_steps_ahead, 2*config1.labels])
                                #print("--------------------------------- X shape--------------------------------=", X.shape)
                                train_RMSE, train_pred, tf_global_step, _, _ = session.run([model.RMSE, model.predictions, model.global_step, model.train_op_en, model.train_op_dec], feed_dict= {model.driving_series: X, model.past_history: Y, model.alphaAtt: alpha}) 



                            epoch_batches += 1

                            accumulated_RMSE += train_RMSE


                            y_true = Y[:, -(config1.nb_steps_ahead):, :]
                            epoch_train_y_true += np.mean(y_true, axis=0)
                            epoch_train_y_pred += np.mean(train_pred, axis=0)
                            epoch_train_x += np.mean(X, axis = 0)

                        except tf.errors.OutOfRangeError:
                            break 
                            
                        
                    epochs_train_y_true.append(epoch_train_y_true/epoch_batches)
                    epochs_train_y_pred.append(epoch_train_y_pred/epoch_batches)
                    epochs_train_x.append(epoch_train_x/epoch_batches)

                    print("train epoch batches=", epoch_batches)

                    best_RMSE_train_per_epoch = float("inf") 


                    total_time = time.time() - initial_time
                    steps_per_second = (tf_global_step - init_global_step) / total_time  
         

                    RMSE_train = accumulated_RMSE / (epoch_batches) 

                    print("[{}] RMSE_train={:.5f}, steps/s={:.5f}".format(tf_global_step, RMSE_train, steps_per_second)) 
                   
                    accumulated_RMSE = 0.0


                    if RMSE_train < best_RMSE_train_per_epoch:  
                        best_RMSE_train_per_epoch = RMSE_train
                        
                    epochs_RMSE_train.append(best_RMSE_train_per_epoch)



                    if (run_validation == True):

                        best_RMSE_val_per_epoch = float("inf") 
                        best_qtc2d_dist_val_per_epoch = float("inf")
                        best_qtc1d_dist_val_per_epoch = float("inf")

                        session.run(val_iterator_x.initializer)
                        session.run(val_iterator_y.initializer)

                        val_scores = model.evaluate(session, val_next_element_x, val_next_element_y, alpha_update )

                        epochs_val_y_true.append(val_scores["y_true_epoch_eval"])
                        epochs_val_y_pred.append(val_scores["y_pred_epoch_eval"])
                        epochs_val_x.append(val_scores["x"])

                        
                        if val_scores["RMSE"] < best_RMSE_val_per_epoch: 
                            best_RMSE_val_per_epoch = val_scores["RMSE"]
                        

                        if val_scores["RMSE"] < best_RMSE: # best RMSE is over all epochs
                            best_RMSE = val_scores["RMSE"]

                        writer.add_summary(make_summary({"min RMSE over val set over all epochs= ": best_RMSE}), tf_global_step)
                        print("----------------------")
                        print("val RMSE of 1 epoch: {:.5f}".format(val_scores["RMSE"]))
                        print("val MAE of 1 epoch: {:.5f}".format(val_scores["MAE"]))
                        print("val MAPE of 1 epoch: {:.5f}".format(val_scores["MAPE"]))
                        print("best_RMSE val over all past epochs={:.5f}".format(best_RMSE))


                        epochs_RMSE_val.append(best_RMSE_val_per_epoch)


            ##################################################################### Post-Processing validation ########################################################################
            

            if (run_validation==True):

 

                print("RMSE_val over epochs len=", len(epochs_RMSE_val))
                plt.figure()
                x = np.linspace(0, config1.num_epochs, num=config1.num_epochs, endpoint=False)
                plt.plot(x, np.asarray(epochs_RMSE_val), label="RMSE over epochs")
                plt.legend(loc='upper right')
                plt.ylabel("RMSE_val")
                plt.xlabel("epochs")
                plt.savefig(log_path / "RMSE_val.png", dpi=300)
                plt.close()



            ##################################################################### Post-Processing training  ########################################################################

            
            if (run_train==True):

                print("Starting post-processing the training results")


                plt.figure()
                x = np.linspace(0, config1.num_epochs, num=config1.num_epochs, endpoint=False)
                plt.plot(x, np.asarray(epochs_RMSE_train), label="train rmse")
                plt.plot(x, np.asarray(epochs_RMSE_val), label="val rmse")
                plt.legend(loc='upper right')
                plt.ylabel("Loss", fontsize=15)
                plt.xlabel("Epochs", fontsize=15)
                plt.savefig(log_path / "RMSE_train_val.png", dpi=300)
                plt.close()



            if (run_test == True):
                print("==================================================Test==================================================")
                for i in range(1):

                    session.run(test_iterator_x.initializer)
                    session.run(test_iterator_y.initializer)
                    test_scores = model.evaluate(session, test_next_element_x, test_next_element_y, alpha_update)


                    if test_scores["RMSE"] < best_RMSE:
                        best_RMSE = test_scores["RMSE"]

                    writer.add_summary(make_summary({"min RMSE = ": best_RMSE}), tf_global_step)
                    print("----------------------")
                    print("RMSE: {:.5f}".format(test_scores["RMSE"]))
                    print("MAE: {:.5f}".format(test_scores["MAE"]))
                    print("MAPE: {:.5f}".format(test_scores["MAPE"]))
                    print("ADE: {:.5f}".format(test_scores["ADE"]))
                    print("AFDE: {:.5f}".format(test_scores["AFDE"]))
                    print("DE-STD: {:.5f}".format(test_scores["DE-STD"]))
                    print("FDE-STD: {:.5f}".format(test_scores["FDE-STD"]))
                    print("best_RMSE={:.5f}".format(best_RMSE))
     


if __name__ == '__main__':
    tf.compat.v1.app.run(main=main)
