'''
#################### Estimators Class ####################

    Author: 
        Tahn Thawainin, AU GAVLAB

    Description: 
        A class that houses various filters/estimators. 
        Filters include:
            *Batch Least Squares
            *Recursive Least Squares / Nonlinear RLS
        {add filters}

    Dependencies:
        <numpy>
        <control>/<control.matlab>

###########################################################
'''
import numpy as np
import torch

class Estimators:

    def __init__(self, n:int, m:int):
        '''
        Class Description:
            A class that houses various filters/estimators. 
        Class Instance Input(s):
            n:  Number of states
                type: <int>
            m:  Number of measurements
                type: <int>
        '''

        assert type(n) == int,\
            f"Input <n> has invalid type. Expected <class 'int'> but recieved {type(n)}"
        assert n > 0,\
            f"Input <n> must be greater than 0. Recieved {n}"
        assert type(m) == int,\
            f"Input <m> has invalid type. Expected <class 'int'> but recieved {type(m)}"
        assert m > 0,\
            f"Input <m> must be greater than 0. Recieved {m}"

        self.n = n
        self.m = m


    # ----- Least Squares -----#
    def ls(self, **kwargs):
        '''
        Descrption:
            Batch least-squares estimation
        Input(s):
            Y:  (m x 1) Measurement value/vector
            H:  (m x n) Observation matrix 
        Output(s):
            x:  (n x 1) State(s) estimate(s)
        '''

        #Declare arguements
        Y = kwargs['Y']
        H = kwargs['H']

        #Assert valid input(s) type
        assert np.asmatrix(Y).shape[0] == self.m and np.asmatrix(Y).shape[1] == 1,\
            f"Input <Y> has invalid dimensions. Expected ({self.m},{1}) but recieved ({np.asmatrix(Y).shape[0]},{np.asmatrix(Y).shape[1]})"
        assert np.asmatrix(H).shape[0] == self.m and np.asmatrix(H).shape[1] == self.n,\
            f"Input <H> has invalid dimensions. Expected ({self.m},{self.n}) but recieved ({np.asmatrix(H).shape[0]},{np.asmatrix(H).shape[1]})"

        #Perform least-squares
        x = np.linalg.inv(np.transpose(H) @ H) @ np.transpose(H) @ Y
        
        return x


    #----- Recursive Least Squares -----#
    def rls(self, nonlinear:bool=False, **kwargs):
        '''
        Descrption: 
            Recursive least-squares estimation. For nonlinear RLS, set nonlinear input to True.
            Algorithm designed to be embedded in an external iterative loop.
        Input(s):
            nonlinear (bool, optional):  Nonlinear RLS configuration. Default is 
            Y:  (m x 1) Measurement value/vector
            H:  (m x n) Linear observation matrix 
            h:  (if NRLS)(m x n) Nonlinear observation matrix 
            R:  (m x m) Measurement covariance matrix
            P:  (n x n) Estimate covariance matrix
            x:  (n x 1) State(s) estimate(s)
        Output(s):
            x:  (n x 1) State(s) estimate(s)
            P:  (n x n) Estimate covariance matrix
            L:  (n x m) Gain matrix
            innov:  (m x 1) Measurement innovation
        '''

        #Declare arguements
        if (nonlinear): h = kwargs['h']
        Y = kwargs['Y']
        H = kwargs['H']
        R = kwargs['R']
        P = kwargs['P']
        x = kwargs['x']
        
        #Assert valid input(s) shape
        if (nonlinear):
            assert np.asmatrix(h).shape[0] == self.m and np.asmatrix(h).shape[1] == self.n,\
            f"Input <h> has invalid dimensions. Expected ({self.m},{self.n}) but recieved ({np.asmatrix(h).shape[0]},{np.asmatrix(h).shape[1]})" 
        assert np.asmatrix(Y).shape[0] == self.m and np.asmatrix(Y).shape[1] == 1,\
            f"Input <Y> has invalid dimensions. Expected ({self.m},{1}) but recieved ({np.asmatrix(Y).shape[0]},{np.asmatrix(Y).shape[1]})"
        assert np.asmatrix(H).shape[0] == self.m and np.asmatrix(H).shape[1] == self.n,\
            f"Input <H> has invalid dimensions. Expected ({self.m},{self.n}) but recieved ({np.asmatrix(H).shape[0]},{np.asmatrix(H).shape[1]})" 
        assert np.asmatrix(R).shape[0] == self.m and np.asmatrix(R).shape[1] == self.m,\
            f"Input <R> has invalid dimensions. Expected ({self.m},{self.m}) but recieved ({np.asmatrix(R).shape[0]},{np.asmatrix(R).shape[1]})"
        assert np.asmatrix(P).shape[0] == self.n and np.asmatrix(P).shape[1] == self.n,\
            f"Input <P> has invalid dimensions. Expected ({self.n},{self.n}) but recieved ({np.asmatrix(P).shape[0]},{np.asmatrix(P).shape[1]})"
        assert np.asmatrix(x).shape[0] == self.n and np.asmatrix(x).shape[1] == 1,\
            f"Input <x> has invalid dimensions. Expection ({self.n},{1}) but revieved ({np.asmatrix(x).shape[0]},{np.asmatrix(x).shape[1]})"

        #Recursive least squares
        L = P @ np.transpose(H) @ np.linalg.inv(H @ P @ np.transpose(H) + R)        #Gain update
        if (nonlinear):
            innov = (Y - h @ x)                                                     #Nonlinear meas innovation
        else:
            innov = (Y - H @ x)                                                     #Measurement innovation
        x = x + L @ (innov)                                                         #State update
        P = (np.identity(self.n) - L @ H) @ P                                       #State covariance update

        return x, P, L, innov
    
    
    #----- Kalman Filter -----#
    def kf(self, T, num_inputs:int=0, as_tensors:bool=False, device='cpu', **kwargs):
        '''
        Description:
            Kalman filter estimation.
            Algorthm designed to be embedded in an external iterative loop.
        Input(s):
            T:  sampling rate
            num_inputs (int):  Number of inputs. Default is 0.
            as_tensors (bool,optional): Tensor conditional. Default is False.
            device (str, optional): If `as_tensors` is passed, the torch device must be passed as well. Default is 'cpu'.
            F:  (n x n) State transition matrix
            B:  (n x a) Input matrix
            u:  (a x 1) State transition input(s)
            Q:  (n x n) Process covariance matrix
            z:  (m x 1) Measurement value/vector
            H:  (m x n) Observation matrix
            R:  (m x m) Measurement covariance matrix
            P:  (n x n) Estimate covariance matrix
            x:  (n x 1) State(s) estimate(s)
        Ouput(s):
            x:  (n x 1) State(s) estimate(s)
            P:  (n x n) Esimate covariance matrix
            K:  (n x m) Kalman gain
            innov:  (m x 1) Measurement innovation  
        '''

        #Declare arguements
        F = kwargs['F']
        if (num_inputs > 0):
            B = kwargs['B']
            u = kwargs['u']
        else:
            B = np.zeros((self.n,1))
            u = np.array([0])
        Q = kwargs['Q']
        z = kwargs['z']
        H = kwargs['H']
        R = kwargs['R']
        P = kwargs['P']
        x = kwargs['x']

        #Convert to tensors if prompted
        if as_tensors:
            if type(F) != torch.Tensor: F = torch.from_numpy(F)
            if type(B) != torch.Tensor: B = torch.from_numpy(B)
            if type(u) != torch.Tensor: u = torch.from_numpy(u)
            if type(Q) != torch.Tensor: Q = torch.from_numpy(Q)
            if type(z) != torch.Tensor: z = torch.from_numpy(z)
            if type(H) != torch.Tensor: H = torch.from_numpy(H)
            if type(R) != torch.Tensor: R = torch.from_numpy(R)
            if type(P) != torch.Tensor: P = torch.from_numpy(P)
            if type(x) != torch.Tensor: x = torch.from_numpy(x)

        #Assert valid input(s) shape
        if as_tensors:
            assert F.shape[0] == self.n and F.shape[1] == self.n,\
                f"Input <F> has invalid dimensions. Expected ({self.n},{self.n}) but recieved ({F.shape[0]},{F.shape[1]})"
            assert B.shape[0] == self.n and B.shape[1] == num_inputs,\
                f"Input <B> has invalid dimensions. Expected ({self.n},{num_inputs}) but recieved ({B.shape[0]},{B.shape[1]})"
            assert u.shape[0] == num_inputs and u.shape[1] == 1,\
                f"Input <u> has invalid dimensions. Expected ({num_inputs},{1}) but recieved ({u.shape[0]},{u.shape[1]})"
            assert z.shape[0] == self.m and z.shape[1] == 1,\
                f"Input <z> has invalid dimensions. Expected ({self.m},{1}) but recieved ({z.shape[0]},{z.shape[1]})"
            assert H.shape[0] == self.m and H.shape[1] == self.n,\
                f"Input <H> has invalid dimensions. Expected ({self.m},{self.n}) but recieved ({H.shape[0]},{H.shape[1]})" 
            assert R.shape[0] == self.m and R.shape[1] == self.m,\
                f"Input <R> has invalid dimensions. Expected ({self.m},{self.m}) but recieved ({R.shape[0]},{R.shape[1]})"
            assert P.shape[0] == self.n and P.shape[1] == self.n,\
                f"Input <P> has invalid dimensions. Expected ({self.n},{self.n}) but recieved ({P.shape[0]},{P.shape[1]})"
            assert x.shape[0] == self.n and x.shape[1] == 1,\
                f"Input <x> has invalid dimensions. Expection ({self.n},{1}) but revieved ({x.shape[0]},{x.shape[1]})"
        else:
            assert np.asmatrix(F).shape[0] == self.n and np.asmatrix(F).shape[1] == self.n,\
                f"Input <F> has invalid dimensions. Expected ({self.n},{self.n}) but recieved ({np.asmatrix(F).shape[0]},{np.asmatrix(F).shape[1]})"
            assert np.asmatrix(B).shape[0] == self.n and np.asmatrix(B).shape[1] == num_inputs,\
                f"Input <B> has invalid dimensions. Expected ({self.n},{num_inputs}) but recieved ({np.asmatrix(B).shape[0]},{np.asmatrix(B).shape[1]})"
            assert np.asmatrix(u).shape[0] == num_inputs and np.asmatrix(u).shape[1] == 1,\
                f"Input <u> has invalid dimensions. Expected ({num_inputs},{1}) but recieved ({np.asmatrix(u).shape[0]},{np.asmatrix(u).shape[1]})"
            assert np.asmatrix(z).shape[0] == self.m and np.asmatrix(z).shape[1] == 1,\
                f"Input <z> has invalid dimensions. Expected ({self.m},{1}) but recieved ({np.asmatrix(z).shape[0]},{np.asmatrix(z).shape[1]})"
            assert np.asmatrix(H).shape[0] == self.m and np.asmatrix(H).shape[1] == self.n,\
                f"Input <H> has invalid dimensions. Expected ({self.m},{self.n}) but recieved ({np.asmatrix(H).shape[0]},{np.asmatrix(H).shape[1]})" 
            assert np.asmatrix(R).shape[0] == self.m and np.asmatrix(R).shape[1] == self.m,\
                f"Input <R> has invalid dimensions. Expected ({self.m},{self.m}) but recieved ({np.asmatrix(R).shape[0]},{np.asmatrix(R).shape[1]})"
            assert np.asmatrix(P).shape[0] == self.n and np.asmatrix(P).shape[1] == self.n,\
                f"Input <P> has invalid dimensions. Expected ({self.n},{self.n}) but recieved ({np.asmatrix(P).shape[0]},{np.asmatrix(P).shape[1]})"
            assert np.asmatrix(x).shape[0] == self.n and np.asmatrix(x).shape[1] == 1,\
                f"Input <x> has invalid dimensions. Expection ({self.n},{1}) but revieved ({np.asmatrix(x).shape[0]},{np.asmatrix(x).shape[1]})"
            
        if as_tensors:
            # Time update
            if num_inputs <= 1:
                x = F @ x + B*u                 #State propagation
            else:
                x = F @ x + B @ u                  
            P = F @ P @ torch.t(F) + Q     #Priori covariance update

            # Measurement update
            K = P @ torch.t(H) @ torch.linalg.inv(H @ P @ torch.t(H) + R)    #Kalman gain
            innov = (z - H @ x)                                                     #Innovation
            x = x + K @ (innov)
            I = torch.eye(self.n).to(device=device, dtype=torch.float32)                                                     #State correction
            P = (I - K @ H) @ P
            
        else:                                #Covariance update
            # Time update
            if num_inputs <= 1:
                x = F @ x + B*u                 #State propagation
            else:
                x = F @ x + B @ u                  
            P = F @ P @ np.transpose(F) + Q     #Priori covariance update

            # Measurement update
            K = P @ np.transpose(H) @ np.linalg.inv(H @ P @ np.transpose(H) + R)    #Kalman gain
            innov = (z - H @ x)                                                     #Innovation
            x = x + K @ (innov)                                                     #State correction
            P = (np.identity(self.n) - K @ H) @ P                                   #Covariance update

        return x, P, K, innov
        

    #----- Extended Kalman Filter -----#
    '''
    ***INSERT DESCRIPTION HERE***

    '''
    def ekf(self):
        pass