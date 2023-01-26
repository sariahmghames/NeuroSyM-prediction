import numpy as np
import tensorflow as tf

from tensorflow.python.layers.core import dense
from tensorflow.python.ops.rnn_cell_impl import LSTMCell

from config import Config

from math import sqrt



tf.executing_eagerly() 


class TimeAttnModel:

    def __init__(self, config: Config, alpha_update: False, inp_att:False, temp_att:False):
        self.config = config
        self.alpha_update = alpha_update
        self.inp_att_enabled = inp_att
        self.temporal_att_enabled = temp_att

        self.driving_series = tf.compat.v1.placeholder(tf.float32, [self.config.batch_size,
                                                          2*self.config.n,  # n (number of supporting series)
                                                          self.config.T,
                                                          ])  # T (length of a time window)    
        self.past_history = tf.compat.v1.placeholder(tf.float32, [self.config.batch_size,
                                                        self.config.T-1+self.config.nb_steps_ahead, 
                                                        2*self.config.labels])  



        self.alphaAtt = tf.compat.v1.placeholder(tf.float32, [self.config.batch_size,
                                                        2*self.config.n, # T
                                                        self.config.T])  


        self.predictions, self.loss = self.get_predictions_and_loss(self.driving_series,
                                                                    self.past_history)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.reset_global_step = tf.compat.v1.assign(self.global_step, 0)
        learning_rate = tf.compat.v1.train.exponential_decay(self.config.learning_rate, self.global_step,
                                                   self.config.decay_frequency, self.config.decay_rate,
                                                   staircase=True) 

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


        self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tf.reshape(self.past_history[:, -config.nb_steps_ahead:, :], [-1]), 
                                                                 tf.reshape(self.predictions, [-1]))))) 

        self.MAE = tf.reduce_mean( # mean absolute error
            tf.abs(
                tf.subtract(tf.reshape(self.past_history[:, -config.nb_steps_ahead:,:], [-1]), tf.reshape(self.predictions, [-1])))) 

        self.MAPE = tf.reduce_mean( # mean absolute percentage error
            tf.abs(
                tf.divide(
                    tf.subtract(tf.reshape(self.past_history[:, -config.nb_steps_ahead:,:], [-1]), tf.reshape(self.predictions, [-1])),
                    tf.reshape(self.past_history[:, -config.nb_steps_ahead:,:], [-1])
                )
            )
        ) * 100

        self.DE = tf.math.sqrt(tf.math.square(tf.subtract(tf.reshape(self.past_history[:, -config.nb_steps_ahead:,:self.config.labels], [-1]), tf.reshape(self.predictions[:, :, :self.config.labels], [-1]))) + tf.math.square(tf.subtract(tf.reshape(self.past_history[:, -config.nb_steps_ahead:,self.config.labels:], [-1]), tf.reshape(self.predictions[:,:, self.config.labels:], [-1]))))

        
        self.FDE = tf.math.sqrt(tf.math.square(tf.subtract(self.past_history[:, -1 ,:self.config.labels],self.predictions[:, -1, :self.config.labels])) + tf.math.square(tf.subtract(self.past_history[:, -1 ,self.config.labels:], self.predictions[:,-1, self.config.labels:])))

        self.ADE = tf.reduce_mean(self.DE)

        
        self.AFDE = tf.reduce_mean(self.FDE)


    def _attention(self, hidden_state, cell_state, input):
        print(" _attention ===================================================== ")
        attn_input = tf.concat([hidden_state, cell_state], axis=1)  
        attn_input = tf.reshape(tf.tile(attn_input, [1, input.shape[1]]),  
                                [self.config.batch_size, input.shape[1], 2 * hidden_state.shape[1]] 
        )
        z = tf.tanh(dense(attn_input, input.shape[2]) + dense(input, input.shape[2], use_bias=False)) 

        pre_softmax_attn = tf.compat.v1.layers.dense(z, 1) 
        return tf.nn.softmax(pre_softmax_attn) 




    def _attention_enc(self, hidden_state, cell_state, input, t):
        attn_input = tf.concat([hidden_state, cell_state], axis=1)  
        attn_input = tf.reshape(tf.tile(attn_input, [1, input.shape[1]]),  
                                [self.config.batch_size, input.shape[1], 2 * hidden_state.shape[1]] 
        )

        alpha = tf.expand_dims(self.alphaAtt[:, :, t], axis = -1)


        z = tf.tanh(dense(attn_input, input.shape[2]) + dense(input, input.shape[2], use_bias=False)) 

        pre_softmax_attn = tf.compat.v1.layers.dense(z, 1) 
        pre_softmax_attn = alpha * pre_softmax_attn 
        return tf.nn.softmax(pre_softmax_attn) 





    def get_predictions_and_loss(self, driving_series, past_history):

        # define encoder
        with tf.compat.v1.variable_scope("EncoderRNN"):
            cell = LSTMCell(self.config.m, state_is_tuple=True) 
            initial_state = cell.zero_state(self.config.batch_size, tf.float32) 
            state = initial_state 
            s, h = state
            outputs = []
            outputs_decoder = []
            y_Ts = []

            for t in range(self.config.T):
                if self.inp_att_enabled:
                    print(" inp att enabled ===================================================== ")

                    if (self.alpha_update == True):
                        alpha = self._attention_enc(h, s, driving_series,t)
                    else:
                        alpha = self._attention(h, s, driving_series) 
                   

                    x_tilde = tf.squeeze(alpha) * driving_series[:, :, t] 
                else:
                    print(" inp att not enabled ===================================================== ")
                    x_tilde = driving_series[:, :, t] 


                (cell_output, state) = cell(x_tilde, state)  
                s, h = state
                outputs.append(h) 

        encoder_outputs = tf.concat(outputs, axis=1) # along columns
        encoder_outputs = tf.reshape(encoder_outputs, [self.config.batch_size, self.config.T, -1]) 
        

        # define decoder
        with tf.compat.v1.variable_scope("DecoderRNN"):
            cell = LSTMCell(self.config.p, state_is_tuple=True) 
            initial_state = cell.zero_state(self.config.batch_size, tf.float32)
            c_t = tf.compat.v1.get_variable("c_t", [self.config.batch_size, self.config.m]) 
            state = initial_state
            s_, d = state

            for t in range(self.config.T):
                if self.temporal_att_enabled:
                    print(" temp att enabled ===================================================== ")
                    beta = self._attention(d, s_, encoder_outputs)
                    c_t = tf.reduce_sum(beta * encoder_outputs, axis=1) 
                    c_t = tf.reshape(c_t, [self.config.batch_size, self.config.m])
                else:
                    print(" temp att not enabled ===================================================== ")
                    c_t = encoder_outputs[:, t, :]

                if t < (self.config.T - 1): 
                    y_c = tf.concat([past_history[:, t, :], c_t], axis=1) 
                    

                    y_tilde = tf.compat.v1.layers.dense(y_c, 2* self.config.labels) 
                    (cell_output, state) = cell(y_tilde, state) 
                    s_, d = state
                    outputs_decoder.append(cell_output)

            d_c = tf.concat([d, c_t], axis=1) 
            print("d_c shape=", d_c.shape) 
            y_T = tf.compat.v1.layers.dense(tf.compat.v1.layers.dense(d_c, self.config.p), 2*self.config.labels)  
            print("y_T shape=", y_T.shape) 
            y_Ts.append(y_T)


            ## Add sequential lstm cells 
            for tp in range(self.config.nb_steps_ahead-1):
                y_seq = y_Ts[tp]

                (cell_output, state) = cell(y_seq, state) 
                s_, d = state
                outputs_decoder.append(cell_output)    
                y_T = tf.compat.v1.layers.dense(tf.compat.v1.layers.dense(d, self.config.p), 2*self.config.labels)  
                y_Ts.append(y_T)

            
            y_Ts = tf.squeeze(tf.expand_dims([tf.reshape(a, [self.config.batch_size, 2*self.config.labels]) for a in y_Ts], axis=(0)))
            y_Ts = tf.transpose(y_Ts, (1,0,2))
            print("y_Ts shape =", y_Ts.shape) 


        loss = tf.losses.mean_squared_error(y_Ts, past_history[:, -self.config.nb_steps_ahead:, :]) 
        print("time series cross entropy loss=", loss.shape) 


        return y_Ts, loss


    def evaluate(self, session, next_element_x, next_element_y, alpha_update):
        RMSE_tot = 0.0
        MAE_tot = 0.0
        MAPE_tot = 0.0
        ADE_tot = 0.0
        AFDE_tot = 0.0
        DE_tot = []
        FDE_tot = []


        epoch_batches_y_true = np.zeros((self.config.nb_steps_ahead, 2*self.config.labels))
        epoch_batches_y_pred = np.zeros((self.config.nb_steps_ahead, 2*self.config.labels))
        epoch_x = np.zeros((2*self.config.n, self.config.T))

        loss_tot = 0.0
        acc_tot = 0.0

        num_batches = 0

        while True:
            try:

                if (alpha_update == False):

                    xX, yX= session.run(next_element_x)
                    xY, yY= session.run(next_element_y)

                    X = np.concatenate((xX, xY), axis = 1)
                    Y = np.concatenate((yX, yY), axis = 1)

                    Y = np.reshape(Y, [self.config.batch_size, self.config.T-1+self.config.nb_steps_ahead, 2*self.config.labels])
                    loss_val, pred_val, RMSE_val, MAE_val, MAPE_val, ADE_val, AFDE_val, DE_val, FDE_val  = session.run([self.loss, self.predictions, self.RMSE, self.MAE, self.MAPE, self.ADE, self.AFDE, self.DE, self.FDE ],feed_dict= {self.driving_series: X, self.past_history: Y})


                else:

                    xX, yX, Alpha1 = session.run(next_element_x)
                    xY, yY, Alpha2 = session.run(next_element_y)

                    X = np.concatenate((xX, xY), axis = 1)
                    Y = np.concatenate((yX, yY), axis = 1)
                    alpha = np.concatenate((Alpha1, Alpha2), axis = 1)
                    Y = np.reshape(Y, [self.config.batch_size, self.config.T-1+self.config.nb_steps_ahead, 2*self.config.labels])
                    loss_val, pred_val, RMSE_val, MAE_val, MAPE_val, ADE_val, AFDE_val, DE_val, FDE_val = session.run([self.loss, self.predictions, self.RMSE, self.MAE, self.MAPE, self.ADE, self.AFDE, self.DE, self.FDE],feed_dict= {self.driving_series: X, self.past_history: Y, self.alphaAtt: alpha})


                num_batches += 1

                loss_tot += np.sum(loss_val)
                RMSE_tot += RMSE_val 
                MAE_tot += MAE_val 
                MAPE_tot += MAPE_val 
                ADE_tot += ADE_val
                AFDE_tot += AFDE_val
                DE_tot.append(DE_val)
                FDE_tot.append(FDE_val)


                y_true = Y[:, -self.config.nb_steps_ahead:, :] 
                epoch_batches_y_true += np.mean(y_true, axis=0) 
                epoch_batches_y_pred += np.mean(pred_val, axis=0) 
                epoch_x += np.mean(X, axis=0)

            except tf.errors.OutOfRangeError:
                break 

        print("nb batches in val=", num_batches) 
        print("evaluate loss shape= ", loss_tot.shape) 

        scores = {}
        scores["x"] = epoch_x/num_batches
        scores["y_true_epoch_eval"] = epoch_batches_y_true/num_batches
        scores["y_pred_epoch_eval"] = epoch_batches_y_pred/num_batches
        scores["num_batches"] = num_batches
        scores["loss_val"] = loss_tot / (num_batches) 
        scores["RMSE"] = RMSE_tot / (num_batches ) 
        scores["MAE"] = MAE_tot / (num_batches )
        scores["MAPE"] = MAPE_tot / (num_batches)
        scores["ADE"] = ADE_tot / (num_batches )
        scores["AFDE"] = AFDE_tot / (num_batches)
        scores["DE-STD"] = np.std(tf.concat(DE_tot, axis=0).eval())
        scores["FDE-STD"] = np.std(tf.concat(FDE_tot, axis=0).eval())

        return scores
