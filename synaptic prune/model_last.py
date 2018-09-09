# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
from tensorflow.examples.tutorials.mnist import input_data

# variable initialization functions

i_number = 0 ;
def weight_variable(shape):
    with tf.name_scope('weights'): 
        initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    with tf.name_scope('biases'):
        initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Model:
    def __init__(self, x, y_,y1_,y11_,y111_,keep_prob):
        self.learning_rate = 0.05  # 初始学习速率时0.1
        decay_rate = 0.96  # 衰减率
        decay_steps = 100  # 衰减次数
        self.global_ = tf.Variable(tf.constant(0))
        
        
        
        self.learning_rate1 = tf.train.exponential_decay(self.learning_rate, self.global_, decay_steps, decay_rate, staircase=True)
        
        
        
        in_dim = int(x.get_shape()[1]) # 784 for MNIST
        out_dim = int(y_.get_shape()[1]) # 10 for MNIST

        self.x = x # input placeholder
        self.keep_prob = keep_prob
        # simple 2-layer network
        self.W1 = weight_variable([in_dim,16])
        self.b1 = bias_variable([16])

        self.W2 = weight_variable([16,out_dim])
        self.b2 = bias_variable([out_dim])
        
        self.W22 = weight_variable([16,out_dim])
        self.b22 = weight_variable([out_dim])
        
        self.W222 = weight_variable([16,out_dim])
        self.b222 = weight_variable([out_dim]) 
        
        self.W2222 = weight_variable([16,out_dim])
        self.b2222 = weight_variable([out_dim])
        
        
        
  #      with tf.name_scope('Wx_plus_b'):  
        h1 = tf.nn.relu(tf.matmul(x,self.W1) + self.b1) # hidden layer
        h11 = tf.nn.dropout(h1,keep_prob)
        
        self.y = tf.matmul(h11,self.W2) + self.b2 # output layer
        self.y1 = tf.matmul(h11,self.W22) + self.b22 
        self.y11 = tf.matmul(h11,self.W222) +self.b222
        self.y111 = tf.matmul(h11,self.W222)+self.b2222

        self.var_list = [self.W1, self.b1, self.W2, self.b2]
        self.var_list_output = [self.W1,self.b1,self.W22,self.b22]
        self.var_list_output1 = [self.W1,self.b1,self.W222,self.b222]
        self.var_list_output2 = [self.W1,self.b1,self.W2222,self.b2222]
        

        # vanilla single-task loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        
        
        self.cross_entropy_output = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y1_,logits=self.y1))
        
        self.cross_entropy_output1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y11_,logits=self.y11))
        
        self.cross_entropy_output2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y111_,logits=self.y111));
        
        self.set_vanilla_loss()
        self.set_vanilla_loss_output()
        self.set_vanilla_loss_output1() 
        self.set_vanilla_loss_output2()
        
        

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        correct_prediction_output = tf.equal(tf.argmax(self.y1,1),tf.argmax(y1_,1))
        correct_prediction_output1 = tf.equal(tf.argmax(self.y11,1),tf.argmax(y11_,1))
        correct_prediction_output2 = tf.equal(tf.argmax(self.y111,1),tf.argmax(y111_,1))
        
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy_output = tf.reduce_mean(tf.cast(correct_prediction_output,tf.float32))
        self.accuracy_output1 = tf.reduce_mean(tf.cast(correct_prediction_output1,tf.float32))
        self.accuracy_output2 = tf.reduce_mean(tf.cast(correct_prediction_output2,tf.float32))
        
        
        self.F_accum = [] 
        self.last_accum = []
        
        
        
        
        
        

    def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task     
            
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(self.F_accum)
            mean_diffs = np.zeros(0)

        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1],self.keep_prob:1})
           
            # square the derivatives and add to total
            
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(self.F_accum)):
                        F_diff += np.sum(np.absolute(self.F_accum[v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    for v in range(len(self.F_accum)):
                        F_prev[v] = self.F_accum[v]/(i+1)
                    plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
                    plt.xlabel("Number of samples")
                    plt.ylabel("Mean absolute Fisher difference")
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples
            
            
   
        
            
    #第二个        
    def compute_fisher_output(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task     
            
        self.F_accum = []
        for v in range(len(self.var_list_output)):
            self.F_accum.append(np.zeros(self.var_list_output[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y1)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(self.F_accum)
            mean_diffs = np.zeros(0)

        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list_output), feed_dict={self.x: imgset[im_ind:im_ind+1],self.keep_prob:1})
           
            # square the derivatives and add to total
            
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(self.F_accum)):
                        F_diff += np.sum(np.absolute(self.F_accum[v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    for v in range(len(self.F_accum)):
                        F_prev[v] = self.F_accum[v]/(i+1)
                    plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
                    plt.xlabel("Number of samples")
                    plt.ylabel("Mean absolute Fisher difference")
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples
            
            
     #第三个
    def compute_fisher_all(self,imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task     
            
        self.F_accum = []
        for v in range(len(self.var_list_output1)):
            self.F_accum.append(np.zeros(self.var_list_output1[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y11)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(self.F_accum)
            mean_diffs = np.zeros(0)

        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]),self.var_list_output1), feed_dict={self.x: imgset[im_ind:im_ind+1],self.keep_prob:1})
    
           
            # square the derivatives and add to total
            
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(self.F_accum)):
                        F_diff += np.sum(np.absolute(self.F_accum[v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    for v in range(len(self.F_accum)):
                        F_prev[v] = self.F_accum[v]/(i+1)
                    plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
                    plt.xlabel("Number of samples")
                    plt.ylabel("Mean absolute Fisher difference")
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples      
     #第四个
    
    def compute_fisher_output2(self,imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task     
            
        self.F_accum = []
        for v in range(len(self.var_list_output2)):
            self.F_accum.append(np.zeros(self.var_list_output2[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y111)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(self.F_accum)
            mean_diffs = np.zeros(0)

        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]),self.var_list_output2), feed_dict={self.x: imgset[im_ind:im_ind+1],self.keep_prob:1})
    
           
            # square the derivatives and add to total
            
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(self.F_accum)):
                        F_diff += np.sum(np.absolute(self.F_accum[v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    for v in range(len(self.F_accum)):
                        F_prev[v] = self.F_accum[v]/(i+1)
                    plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
                    plt.xlabel("Number of samples")
                    plt.ylabel("Mean absolute Fisher difference")
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples   
            
            
            
    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())
            
    def star_output(self):
        # used for saving optimal weights after most recent task training
        self.star_vars_output = []

        for v in range(len(self.var_list_output)):
            self.star_vars_output.append(self.var_list_output[v].eval())
    def star_output1(self):
        # used for saving optimal weights after most recent task training
        self.star_vars_output1 = []

        for v in range(len(self.var_list_output1)):
            self.star_vars_output1.append(self.var_list_output1[v].eval())
    
    def star_output2(self):
        # used for saving optimal weights after most recent task training
        self.star_vars_output2 = []

        for v in range(len(self.var_list_output2)):
            self.star_vars_output2.append(self.var_list_output2[v].eval())
            
            

    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))
    def restore_output(self,sess):
        if hasattr(self,"star_vars_output"):
            for v in range(len(self.var_list_output)):
                sess.run(self.var_list_output[v].assign(self.star_vars_output[v]))
    def restore_output1(self,sess):
        if hasattr(self,"star_vars_output1"):
            for v in range(len(self.var_list_output1)):
                sess.run(self.var_list_output1[v].assign(self.star_vars_output1[v])) 
    def restore_output2(self,sess):
        if hasattr(self,"star_vars_output2"):
            for v in range(len(self.var_list_output2)):
                sess.run(self.var_list_output2[v].assign(self.star_vars_output2[v])) 
    

    def set_vanilla_loss(self):
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy) 
        
    def set_vanilla_loss_output(self):
        self.train_step_output = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy_output)
    def set_vanilla_loss_output1(self):
        self.train_step_output1 = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy_output1) 
        
    def set_vanilla_loss_output2(self):
        self.train_step_output2 = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy_output2) 
        
    def get_next_F_accum(self):
        for v in range(len(self.F_accum)-1):
            self.F_accum[v] += self.last_accum[v] 
            self.last_accum[v] = self.F_accum[v]
            
    def get_F_accum(self):
        for v in range(len(self.F_accum)-1):
            self.F_accum[v] += self.last_accum[v]
            
            
    def save_accum(self):
        self.last_accum = [0,0,0,0]
        for v in range(len(self.F_accum)):
            self.last_accum[v] = self.F_accum[v]
    def save_last_accum(self):
        for v in range(len(self.F_accum)-1):
            self.last_accum[v] = self.F_accum[v]
        
            
            
    def save_all(self):
        self.F_accum_temp = deepcopy(self.F_accum)
        self.var_list_temp = [0,0,0,0]
        for i in range(len(self.var_list)):
            self.var_list_temp[i] = tf.multiply(self.var_list[i],1) 
        
    def get_all(self):
        self.F_accum = deepcopy(self.F_accum_temp)
        sess.run(self.W1.assign(self.var_list_temp[0]))
        sess.run(self.b1.assign(self.var_list_temp[1]))
        sess.run(self.W2.assign(self.var_list_temp[2]))
        sess.run(self.b2.assign(self.var_list_temp[3]))
        
            
    def update_ewc_loss(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints
            
     
        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.cross_entropy
 

        for v in range(len(self.var_list)):
            self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(tf.cast(self.F_accum[v],tf.float32),tf.square(self.var_list[v] - self.star_vars[v])))
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.ewc_loss)
        
    def update_ewc_loss_output(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints
            
     
        
        self.ewc_loss = self.cross_entropy_output
 

        for v in range(len(self.var_list)-1):
            self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(tf.cast(self.F_accum[v],tf.float32),tf.square(self.var_list_output[v] - self.star_vars_output[v])))
        self.train_step_ewc_output = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.ewc_loss)

    def update_ewc_loss_output_all(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints
            
     
        
        self.ewc_loss = self.cross_entropy_output1 
 

        for v in range(len(self.var_list_output1)-1):
            self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(tf.cast(self.F_accum[v],tf.float32),tf.square(self.var_list_output1[v] - self.star_vars_output1[v])))
        self.train_step_ewc_output1 = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.ewc_loss)

        
    def update_ewc_loss_output2(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints
            
     
        
        self.ewc_loss = self.cross_entropy_output2 
 

        for v in range(len(self.var_list_output2)-1):
            self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(tf.cast(self.F_accum[v],tf.float32),tf.square(self.var_list_output2[v] - self.star_vars_output2[v])))
        self.train_step_ewc_output2 = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.ewc_loss)        

    
    def or_mask(self,mask1,mask2):
        temp = deepcopy(mask1)
        for i in range(len(mask1)-1):
            for m in range(np.shape(mask1[i])[0]):
                for n in range(np.shape(mask1[i])[1]):
                    if mask1[i][m,n]==1 or mask2[i][m,n]==1:
                        temp[i][m,n] = 1
                    else:
                        temp[i][m,n] = 0 
        return temp 

        



        
        
        
        




