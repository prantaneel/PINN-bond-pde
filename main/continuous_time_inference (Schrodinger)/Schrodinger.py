"""
@author: Maziar Raissi
"""
import os
os.chdir('/Users/prantaneeldebnath/Downloads/PINN-pde/PINN-bond-pde/main/continuous_time_inference (Schrodinger)')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
# from scipy.interpolate import griddata
from pyDOE import lhs
# from plotting import newfig, savefig
# from mpl_toolkits.mplot3d import Axes3D
# import time
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.axes_grid1 import make_axes_locatable

tf.compat.v1.disable_eager_execution()

np.random.seed(1234)
tf.random.set_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub):
        
        X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0) for time 0 so appending the zero time in (x, t) to create X0
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # tb is the list of boundary times we append with the same size of lower bound x to create (-5, t)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # tb is the list of boundary times we append with the same size of upper bound x to create (5, t)
        
        self.lb = lb 
        self.ub = ub
               
        self.x0 = X0[:,0:1] #x component of X0
        self.t0 = X0[:,1:2] #t component of X0

        self.x_lb = X_lb[:,0:1] #x component of X_lb
        self.t_lb = X_lb[:,1:2] #t component of X_lb

        self.x_ub = X_ub[:,0:1]
        self.t_ub = X_ub[:,1:2]
        
        self.x_f = X_f[:,0:1] #x comp of function test points X_f
        self.t_f = X_f[:,1:2] #t comp of function test points X_f
        
        self.u0 = u0 #value of u at t = 0
        self.v0 = v0 #value of v at t = 0
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers) #initialize the weights and biases 
        
        # tf Placeholders        
        #placeholders are similar to variables which are always fed
        self.x0_tf = tf.keras.layers.Input(dtype=tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.keras.layers.Input(dtype=tf.float32, shape=[None, self.t0.shape[1]])
        
        self.u0_tf = tf.keras.layers.Input(dtype=tf.float32, shape=[None, self.u0.shape[1]])
        self.v0_tf = tf.keras.layers.Input(dtype=tf.float32, shape=[None, self.v0.shape[1]])
        
        self.x_lb_tf = tf.keras.layers.Input(dtype=tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.keras.layers.Input(dtype=tf.float32, shape=[None, self.t_lb.shape[1]])
        
        self.x_ub_tf = tf.keras.layers.Input(dtype=tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.keras.layers.Input(dtype=tf.float32, shape=[None, self.t_ub.shape[1]])
        
        self.x_f_tf = tf.keras.layers.Input(dtype=tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.keras.layers.Input(dtype=tf.float32, shape=[None, self.t_f.shape[1]])
        #PLACEHOLDERS are used for creating the control flow graph of the function and can represent nodes
        # tf Graphs
        self.u0_pred, self.v0_pred, _ , _ = self.net_uv(self.x0_tf, self.t0_tf) #returns the u, v for the given data
        self.u_lb_pred, self.v_lb_pred, self.u_x_lb_pred, self.v_x_lb_pred = self.net_uv(self.x_lb_tf, self.t_lb_tf)
        self.u_ub_pred, self.v_ub_pred, self.u_x_ub_pred, self.v_x_ub_pred = self.net_uv(self.x_ub_tf, self.t_ub_tf)
        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf) #instead of directly returning the data, we return the function
        
        # Loss
        #loss function calculates the loss for each component
        #loss function need to be defined carefully
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.v0_tf - self.v0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.v_lb_pred - self.v_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.v_x_lb_pred - self.v_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred))
        
        # Optimizers
        #we define the custom scipy interface optimizer with the self.loss as the loss function
        # self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
        #                                                         method = 'L-BFGS-B', 
        #                                                         options = {'maxiter': 50000,
        #                                                                    'maxfun': 50000,
        #                                                                    'maxcor': 50,
        #                                                                    'maxls': 50,
        #                                                                    'ftol' : 1.0 * np.finfo(float).eps})
        var_list = [self.weights, self.biases]
        self.optimizer_Adam = tf.keras.optimizers.Adam(0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, var_list=var_list) #use the ADAM optimizer to optimize the loss function
                
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

#SESS.RUN RUNS THE GIVEN DAG 
              
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        #calculates the neural network output for a given input
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_uv(self, x, t):
        #creates a neural net for a given (x, t) as input
        X = tf.concat([x,t],1)
        #X is a matrix which has 2 inputs and is a matrix of the size Px2; P = length of the input dataset
        
        uv = self.neural_net(X, self.weights, self.biases)
        #calculates the uv for the given inputs
        u = uv[:,0:1]
        v = uv[:,1:2]
        
        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]
        #calculates the u and v gradients wrt x
        return u, v, u_x, v_x

    def net_f_uv(self, x, t):
        #returns the value of the function for the given PDE
        u, v, u_x, v_x = self.net_uv(x,t)
        #net_uv returns the values of u and v for a given (x, t) and the respective x gradients
        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        
        v_t = tf.gradients(v, t)[0]
        v_xx = tf.gradients(v_x, x)[0]
        
        f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
        f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u   
        
        return f_u, f_v
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0, self.v0_tf: self.v0,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        #we first state the values that are to be fed to the placeholders
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            #runs the DAG self.train_op_Adam
            #runs the training session using the ADAM optimizer and the tf_dict which specifies the placeholder
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
                                                                                                                          
        # self.optimizer.minimize(self.sess, 
        #                         feed_dict = tf_dict,         
        #                         fetches = [self.loss], 
        #                         loss_callback = self.callback)        
                                    
    
    def predict(self, X_star):
        
        tf_dict = {self.x0_tf: X_star[:,0:1], self.t0_tf: X_star[:,1:2]}
        #we can use any of the net to predict since the underlying structure of the control flow diagram is the same for all (same weights and biases)
        u_star = self.sess.run(self.u0_pred, tf_dict)  
        v_star = self.sess.run(self.v0_pred, tf_dict)  
        #calculates the net for the given data
        
        tf_dict = {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]}
        #change the dict value for the function placeholder and use the net_f_uv to compute the function 
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)
        #return the predicted output and the output of the PDE for each component which can be used for error calculations
        return u_star, v_star, f_u_star, f_v_star
    
if __name__ == "__main__": 
     
    noise = 0.0        
    
    # Doman bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    N0 = 50 #number of points for time = 0
    N_b = 50 #number of points for boundary
    N_f = 20000 #number of simulated solutions to the PDE
    layers = [2, 100, 100, 100, 100, 2] 
    #4 hidden layers with 100 neurons
    #output is u, v solution is u + iv
    #input is x, t
       
    data = scipy.io.loadmat('NLS.mat')
    
    t = data['tt'].flatten()[:,None] #time data indexed
    x = data['x'].flatten()[:,None] #position data indexed [[-1], [0], [1], ...]
    Exact = data['uu'] #exact imaginary simulated points
    #exact is a grid of values for all the t and x points correspondingly
    Exact_u = np.real(Exact) #real part of the exact point
    Exact_v = np.imag(Exact) #imaginary part of the exact point
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2) #modulus of the exact point
    
    # Exact_u of the form [[a1, 0], [a2, 0], [a3, 0], ... ]
    
    X, T = np.meshgrid(x,t)
    #create a mesh grid with the available t, x points pairwise
    # for X, x is constant downwards and changes columnwise-wise
    #for T, its constant columnwise and changes row-wise
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) #[[x1, t1], [x2, t1], [x3, t1], ...., [xN, tM]]
    # concats both arrays across the second axis
    u_star = Exact_u.T.flatten()[:,None] # for each X_star the corresponding output u_star
    v_star = Exact_v.T.flatten()[:,None] # for each X_star the corresponding output v_star
    h_star = Exact_h.T.flatten()[:,None] # for each X_star the corresponding output h_star
    
    ###########################
   
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    #random x indices for N0 size choices 
    x0 = x[idx_x,:] # N0 position values vector
    u0 = Exact_u[idx_x,0:1] #u values only for time 0 for all those x0
    v0 = Exact_v[idx_x,0:1] #v values only for time 0 for all those x0
    
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t,:] #choose N_b values of time from time array
    
    X_f = lb + (ub-lb)*lhs(2, N_f)
    print(X_f)
    #creates random points of size N_f 
    model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub)
             
    start_time = time.time()                
    model.train(50000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
        
    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
        
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_h = np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))

    
#     U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
#     V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
#     H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')

#     FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')
#     FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method='cubic')     
    

    
#     ######################################################################
#     ############################# Plotting ###############################
#     ######################################################################    
    
#     X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
#     X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
#     X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
#     X_u_train = np.vstack([X0, X_lb, X_ub])

#     fig, ax = newfig(1.0, 0.9)
#     ax.axis('off')
    
#     ####### Row 0: h(t,x) ##################    
#     gs0 = gridspec.GridSpec(1, 2)
#     gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
#     ax = plt.subplot(gs0[:, :])
    
#     h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu', 
#                   extent=[lb[1], ub[1], lb[0], ub[0]], 
#                   origin='lower', aspect='auto')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     fig.colorbar(h, cax=cax)
    
#     ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)
    
#     line = np.linspace(x.min(), x.max(), 2)[:,None]
#     ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
#     ax.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
#     ax.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)    
    
#     ax.set_xlabel('$t$')
#     ax.set_ylabel('$x$')
#     leg = ax.legend(frameon=False, loc = 'best')
# #    plt.setp(leg.get_texts(), color='w')
#     ax.set_title('$|h(t,x)|$', fontsize = 10)
    
#     ####### Row 1: h(t,x) slices ##################    
#     gs1 = gridspec.GridSpec(1, 3)
#     gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
#     ax = plt.subplot(gs1[0, 0])
#     ax.plot(x,Exact_h[:,75], 'b-', linewidth = 2, label = 'Exact')       
#     ax.plot(x,H_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$|h(t,x)|$')    
#     ax.set_title('$t = %.2f$' % (t[75]), fontsize = 10)
#     ax.axis('square')
#     ax.set_xlim([-5.1,5.1])
#     ax.set_ylim([-0.1,5.1])
    
#     ax = plt.subplot(gs1[0, 1])
#     ax.plot(x,Exact_h[:,100], 'b-', linewidth = 2, label = 'Exact')       
#     ax.plot(x,H_pred[100,:], 'r--', linewidth = 2, label = 'Prediction')
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$|h(t,x)|$')
#     ax.axis('square')
#     ax.set_xlim([-5.1,5.1])
#     ax.set_ylim([-0.1,5.1])
#     ax.set_title('$t = %.2f$' % (t[100]), fontsize = 10)
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)
    
#     ax = plt.subplot(gs1[0, 2])
#     ax.plot(x,Exact_h[:,125], 'b-', linewidth = 2, label = 'Exact')       
#     ax.plot(x,H_pred[125,:], 'r--', linewidth = 2, label = 'Prediction')
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$|h(t,x)|$')
#     ax.axis('square')
#     ax.set_xlim([-5.1,5.1])
#     ax.set_ylim([-0.1,5.1])    
#     ax.set_title('$t = %.2f$' % (t[125]), fontsize = 10)
    
#     # savefig('./figures/NLS')  
    
