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
            nonlinear:  Nonlinear RLS configuration
                        type: <bool>
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
    def kf(self, a:int=0, **kwargs):
        '''
        Description:
            Kalman filter estimation.
            Algorthm designed to be embedded in an external iterative loop.
        Input(s):
            a:  Number of inputs
            F:  (n x n) State transition matrix
            B:  (n x a) Input matrix
            u:  (a x 1) State transition input(s)
            Q:  (n x n) Process covariance matrix
            Y:  (m x 1) Measurement value/vector
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
        if (a > 0):
            B = kwargs['B']
            u = kwargs['u']
        else:
            B = np.zeros((self.n,1))
            u = 0
        Q = kwargs['Q']
        Y = kwargs['Y']
        H = kwargs['H']
        R = kwargs['R']
        P = kwargs['P']
        x = kwargs['x']

        #Assert valid input(s) shape
        assert np.asmatrix(F).shape[0] == self.n and np.asmatrix(F).shape[1] == self.n,\
            f"Input <F> has invalid dimensions. Expected ({self.n},{self.n}) but recieved ({np.asmatrix(F).shape[0]},{np.asmatrix(F).shape[1]})"
        assert np.asmatrix(B).shape[0] == self.n and np.asmatrix(B).shape[1] == a,\
            f"Input <B> has invalid dimensions. Expected ({self.n},{a}) but recieved ({np.asmatrix(B).shape[0]},{np.asmatrix(B).shape[1]})"
        assert np.asmatrix(u).shape[0] == a and np.asmatrix(u).shape[1] == 1,\
            f"Input <u> has invalid dimensions. Expected ({self.a},{1}) but recieved ({np.asmatrix(u).shape[0]},{np.asmatrix(u).shape[1]})"
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
        
        # Time update
        x = F @ x + B @ u                   #State propagation
        P = F @ P @ np.transpose(F) + Q     #Prior covariance update

        # Measurement update
        K = P @ np.transpose(H) @ np.linalg.inv(H @ P @ np.transpose(H) + R)    #Kalman gain
        innov = (Y - H @ x)                                                     #Innovation
        x = x + K @ (innov)                                                     #State correction
        P = (np.identity(self.n) - K @ H) @ P                                   #Covariance update

        return x, P, K, innov
        

    #----- Extended Kalman Filter -----#
    '''
    ***INSERT DESCRIPTION HERE***

    '''
    def ekf(self):
        pass