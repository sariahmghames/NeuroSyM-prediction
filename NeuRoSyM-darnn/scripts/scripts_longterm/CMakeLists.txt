import numpy as np
import tensorflow as tf

from tensorflow.python.layers.core import dense
from tensorflow.python.ops.rnn_cell_impl import LSTMCell

from config import Config

from math import sqrt

from Ind_qtc import QTC 


#tf.config.run_functions_eagerly(True) # v2.7
#tf.compat.v1.enable_eager_execution()
tf.executing_eagerly() 


class TimeAttnModel:

    def __init__(self, config: Config):
        self.config = config

        self.driving_series = tf.compat.v1.placeholder(tf.float32, [self.config.batch_size,  # batch size
                                                          self.config.n,  # n (number of supporting series)
                                                          self.config.T,
                                                          ])  # T (length of a time window)    # tf.placeholder to define the size and type of initialized global variables, global variables are defined/initialized by the placeholder and then feeded when calling a function
        self.past_history = tf.compat.v1.placeholder(tf.float32, [self.config.batch_size,  # batch size
                                                        self.config.T, # T
                                                        self.config.labels])  



        self.alphaAtt = tf.compat.v1.placeholder(tf.float32, [self.config.batch_size,  # batch size
                                                        self.config.n, # T
                                                        self.config.T])  


        self.predictions, self.loss, self.acc = self.get_predictions_and_loss(self.driving_series,
                                                                    self.past_history)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.reset_global_step = tf.compat.v1.assign(self.global_step, 0)
        learning_rate = tf.compat.v1.train.exponential_decay(self.config.learning_rate, self.global_step,
                                                   self.config.decay_frequency, self.config.decay_rate,
                                                   staircase=True) # exponential decay causes the weights to decay exponentially to 0

        trainable_params_en = tf.compat.v1.trainable_variables(scope="EncoderRNN")
        trainable_params_dec = tf.compat.v1.trainable_variables(scope="DecoderRNN")

        gradients_en = tf.compat.v1.gradients(self.loss, trainable_params_en)
        gradients_dec = tf.compat.v1.gradients(self.loss, trainable_params_dec)

        gradients_en, _ = tf.compat.v1.clip_by_global_norm(gradients_en, self.config.max_gradient_norm)
        gradients_dec, _ = tf.compat.v1.clip_by_global_norm(gradients_dec, self.config.max_gradient_norm)

        optimizers = {
            "adam": tf.compat.v1.train.AdamOptimizer,
            "sgd": tf.compat.v1.train.GradientDescentOptimizer
        }

        optimizer_en = optimizers[self.config.optimizer](learning_rate) # optimizer of encoder
        optimizer_dec = optimizers[self.config.optimizer](learning_rate) # optimizer of decoder

        self.train_op_en = optimizer_en.apply_gradients(zip(gradients_en, trainable_params_en))
        self.train_op_dec = optimizer_dec.apply_gradients(zip(gradients_dec, trainable_params_dec), global_step=self.global_step)

        max_prob_predictions = tf.math.argmax(self.predictions, axis = 2) # (10,11)

        max_predictions = [tf.unstack(max_prob_predictions, num=self.config.batch_size, name='unstack')]
 
        self.max_ind_predictions =  tf.cast(max_predictions, tf.float32)
        self.max_ind_predictions_new =  tf.cast(max_prob_predictions, tf.float32)
        print("max_ind_predictions_new shape=", self.max_ind_predictions_new.shape)

        self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tf.reshape(self.past_history[:, -1, :], [-1]), # [-1] reshapes it into a line, dim here is 5 (batch_size)
                                                                 tf.reshape(self.max_ind_predictions, [-1])))))  # 5 (batch_size) * 436 (class nb) = 2180

        self.MAE = tf.reduce_mean( # mean absolute error
            tf.abs(
                tf.subtract(tf.reshape(self.past_history[:, -1,:], [-1]), tf.reshape(self.max_ind_predictions, [-1]))
            )
        ) 

        self.MAPE = tf.reduce_mean( # mean absolute percentage error
            tf.abs(
                tf.divide(
                    tf.subtract(tf.reshape(self.past_history[:, -1,:], [-1]), tf.reshape(self.max_ind_predictions, [-1])),
                    tf.reshape(self.past_history[:, -1,:], [-1])
                )
            )
        ) * 100


        #self.qtc_dist = self.qtc_diff(self.past_history[:, -1, :], self.max_ind_predictions_new)




    # def qtc_diff(self, qtc_ind_true, qtc_ind_pred):

    #     qtc_dic = QTC()

    #     #qtc_dist = tf.compat.v1.placeholder(tf.int32, shape=[self.config.batch_size, self.config.labels])
    #     qtc_dist = np.zeros([self.config.batch_size, self.config.labels])
    #     qtc_dist_el = tf.constant(0)
    #     print("config batch_size=",self.config.batch_size)
    #     print("qtc ind pred shape 0=", qtc_ind_pred.shape[0])
    #     print("qtc ind pred shape 1=", qtc_ind_pred.shape[1])

    #     qtc_lis = qtc_dic.relations_C2()
    #     qtc_list =  tf.compat.v1.convert_to_tensor(np.asarray(qtc_lis))
    #     #print("len of qtc list=", np.array(qtc_lis).shape)
    #     #print("qtc ind true shape=",qtc_ind_true.shape) # (10,11)
    #     #print("qtc ind true =", tf.cast(qtc_ind_true[0,0], tf.int32)) # tensor

    #     for i in range(qtc_ind_true.shape[0]):
    #         for j in range(qtc_ind_true.shape[1]):
    #             print("in of loop of qtc dist")
    #             index_true = tf.cast(qtc_ind_true[i,j], tf.int32)
    #             index_pred = tf.cast(qtc_ind_pred[i,j], tf.int32)
    #             print("index true=", index_true)
    #             qtc_true = qtc_list[index_true]
    #             qtc_pred = qtc_list[index_pred]
    #             print("qtc true=", qtc_true.shape)
    #             print("qtc pred=", qtc_pred.shape)

    #             #for i in range(qtc_true.shape[0]):
    #             qtc_dist_el = tf.math.reduce_sum(tf.abs(tf.subtract(tf.cast(qtc_true, tf.int32),tf.cast(qtc_pred, tf.int32))))

    #             qtc_dist[i,j] = qtc_dist_el.eval()
    #             qtc_dist_el = tf.constant(0)
    #     print("out of loop of qtc dist")

    #     return qtc_dist



    def _attention(self, hidden_state, cell_state, input):
        attn_input = tf.concat([hidden_state, cell_state], axis=1)   # term [h_t-1; s_t-1] in eq (8) 
        attn_input = tf.reshape(tf.tile(attn_input, [1, input.shape[1]]),   # input.shape[1] = self.config.n ; tiling: This operation creates a new tensor by replicating input multiples times; because the embedding vector in eq (8) is for each driving series (kth)
                                [self.config.batch_size, input.shape[1], 2 * hidden_state.shape[1]] # hidden_state.shape[1] = m or p; 2* because size of s tensor = size of h tensor for 1 example and 1 driving series
        )

        z = tf.tanh(dense(attn_input, input.shape[2]) + dense(input, input.shape[2], use_bias=False)) # input.shape[2] is T  and is the size of output of dense layer, the size of output (y) of a dense layer should be equal to size of b if b exist else size of "Ax"= y

        pre_softmax_attn = tf.compat.v1.layers.dense(z, 1) # up here is eq (8) of paper; 1 is size of output of dense layer, which is the kth embedding element e ?
        return tf.nn.softmax(pre_softmax_attn) # attention weights alpha of enc and beta of dec, eq (9) of paper




    def get_predictions_and_loss(self, driving_series, past_history):

        # define encoder
        with tf.compat.v1.variable_scope("EncoderRNN"):
            cell = LSTMCell(self.config.m, state_is_tuple=True) # m is The size of the hidden state of the encoder; # tried with : self.config.m*self.config.labels

            initial_state = cell.zero_state(self.config.batch_size, tf.float32) # hidden state 2nd dim is batch_size; initial_state is h0 is of dim mxbatch_size
            state = initial_state
            s, h = state
            outputs = []

            for t in range(self.config.T):
                # if t > 0: tf.get_variable_scope().reuse_variables()
                if self.config.inp_att_enabled:
                    alpha = self._attention(h, s, driving_series) # attention weights
                    #print("alpha dim=", alpha.shape)
                    if (self.config.update_alpha == True):
                        alpha = alpha * tf.expand_dims(self.alphaAtt[:, :, t], axis = -1)
                    #print("alphaAtt dim=", self.alphaAtt.shape)
                    #print("new alpha dim=", alpha.shape)

                    # input weighted with attention weights
                    x_tilde = tf.squeeze(alpha) * driving_series[:, :, t] # first dim = nb of batches of datasets or nb of examples? ; 2nd dim is = n ; 3rd dim = T 
                else:
                    x_tilde = driving_series[:, :, t] # no input att

                (cell_output, state) = cell(x_tilde, state) # state on right is prev state; state on left is the next state or current state
                s, h = state
                outputs.append(h) # the encoder LSTM outouts h but the decoder LSTM outputs y

        encoder_outputs = tf.concat(outputs, axis=1) # along columns
        encoder_outputs = tf.reshape(encoder_outputs, [self.config.batch_size, self.config.T, -1]) # -1 means that the length in that dimension is inferred.
        print("encoder outputs shape=", encoder_outputs.shape) # (5, 5,64)
        

        # define decoder
        with tf.compat.v1.variable_scope("DecoderRNN"):
            # TODO: check order of hidden states and cell states
            cell = LSTMCell(self.config.p, state_is_tuple=True)  # tried with : self.config.p*self.config.labels
            initial_state = cell.zero_state(self.config.batch_size, tf.float32)
            c_t = tf.compat.v1.get_variable("c_t", [self.config.batch_size, self.config.m]) # c_t is input to decoder, batch_size x m x T , but c_t is at time t so no 3rd dim
            state = initial_state
            s_, d = state

            for t in range(self.config.T):
                # if t > 0: tf.get_variable_scope().reuse_variables()
                if self.config.temporal_att_enabled:
                    beta = self._attention(d, s_, encoder_outputs)
                    #print("beta shape=", beta.shape) # (5, 5,1)
                    c_t = tf.reduce_sum(beta * encoder_outputs, axis=1) # sum along axis = 1 (time) so eq (14)
                    #print("c_t shape=", c_t.shape) # (5, 64) # means batch_size x config.m for each time instant t
                else:
                    c_t = encoder_outputs[:, t, :]

                if t < self.config.T - 1: # the prediction is the last prediction along T window
                    # y_c = tf.concat([tf.expand_dims(past_history[:, t], -1), c_t], axis=1) # t is T-1 in the paper
                    y_c = tf.concat([past_history[:, t, :], c_t], axis=1) # t is T-1 in the paper
                    #print("past_history at t shape=", past_history[:, t, :].shape) # (5, 814)
                    #print("y_c shape=", y_c.shape) # (5, 878)
                    

                    y_tilde = tf.compat.v1.layers.dense(y_c, self.config.labels) # input to the dense layer, 1 if the coeff of "b" param of the dense layer, y_tilde_t-1
                    #print("y_tilde shape=", y_tilde.shape) # (5, 878)
                    (cell_output, state) = cell(y_tilde, state) # cell(input, state)
                    s_, d = state


            d_c = tf.concat([d, c_t], axis=1) # d is hidden state and c is the input at t = T 
            y_T = tf.compat.v1.layers.dense(tf.compat.v1.layers.dense(d_c, self.config.p), self.config.labels*self.config.classes)  # prediction for all batches at time T; 2 dense layers (eq 22 in paper); self.config.p is the size of bw
            y_T = tf.reshape(y_T, [self.config.batch_size,self.config.labels, self.config.classes]) # (5, 814, 436)
            print("y_T shape=", y_T.shape) # (10,11,444)
            
            #y_T = tf.squeeze(y_T) # convert to one-hot vector
            y_T = tf.nn.softmax(y_T, name = None) # output probabilities
            #print("past_history shape=", past_history[:, -1, :].shape) # (5,814)

        loss = tf.losses.mean_squared_error(y_T, past_history[:, - 1, :]) # use the cross_entropy loss here for multi-feature classification, mean_squared_error is for time_series prediction with no classification
        print("time series cross entropy loss=", loss.shape) # Using 'auto'/'sum_over_batch_size' reduction type.

        acc = tf.compat.v1.metrics.accuracy(past_history[:, -1, :], y_T)  
        print("time series cross entropy acc=", acc)


        return y_T, loss, acc


    def evaluate(self, session, next_element):
        RMSE_tot = 0.0
        MAE_tot = 0.0
        MAPE_tot = 0.0
        qtc_dic = QTC()
        qtc_lis = qtc_dic.relations_C2()
        qtc_list =  tf.compat.v1.convert_to_tensor(np.asarray(qtc_lis))

        epoch_batches_y_true = []
        epoch_batches_y_pred = []

        loss_tot = 0.0
        acc_tot = 0.0

        num_batches = 0

        while True:
            try:
                if (self.config.update_alpha == True):
                    x, y, alpha = session.run(next_element)
                    RMSE, MAE, MAPE, pred_val = session.run([self.RMSE, self.MAE, self.MAPE, self.max_ind_predictions_new],
                                                feed_dict={self.driving_series: x, self.past_history: y, self.alphaAtt: alpha})
                else:
                    x, y = session.run(next_element)
                    num_batches += 1
                    loss_val, acc_val, RMSE_val, MAE_val, MAPE_val, pred_val = session.run([self.loss, self.acc, self.RMSE, self.MAE, self.MAPE, self.max_ind_predictions_new],feed_dict= {self.driving_series: x, self.past_history: y})
                    acc_tot += acc_val
                    loss_tot += np.sum(loss_val)
                    RMSE_tot += (RMSE_val ** 2) * self.config.batch_size
                    MAE_tot += MAE_val * self.config.batch_size
                    MAPE_tot += MAPE_val  * self.config.batch_size


                    y_true = y[:, -1, :]
                    epoch_batches_y_true.append(y_true)
                    epoch_batches_y_pred.append(pred_val)


            except tf.errors.OutOfRangeError:
                break 

        print("nb batches in val=", num_batches) # 637
        print("evaluate loss shape= ", loss_tot.shape) # 10,11

        scores = {} # list or dict
        #scores["qtc2d_dist_val"] = tf.math.reduce_mean(tf.math.reduce_mean((qtc_dist_tot / num_batches), axis = 0)) 
        #scores["qtc1d_dist_val"] = tf.math.reduce_mean((qtc_dist_tot / num_batches), axis = 0) 
        scores["y_true_epoch_val"] = epoch_batches_y_true
        scores["y_pred_epoch_val"] = epoch_batches_y_pred
        scores["num_batches"] = num_batches
        scores["loss_val"] = loss_tot / (num_batches) # already using AUTO so sun_over_batch_size
        scores["acc_val"] = np.mean(np.mean((acc_tot / (num_batches)), axis =0))
        scores["RMSE"] = sqrt(RMSE_tot / (num_batches * self.config.batch_size)) # sqrt here because in the while loop we **2 the RMSE variable
        scores["MAE"] = MAE_tot / (num_batches * self.config.batch_size)
        scores["MAPE"] = MAPE_tot / (num_batches * self.config.batch_size)
        return scores
