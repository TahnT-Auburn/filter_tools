'''
#################### Estimators Class ####################

    Author: 
        Tahn Thawainin, AU GAVLAB

    Description: 
        A class that houses various filters/estimators. 
        Filters include:
            *Batch Least Squares
            *Recursive Least Squares
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
    def ls(self, mc:bool=False, **kwargs):
        '''
        Descrption:
            Batch least-squares estimation
        Input(s):
            mc: Monte Carlo configuration
                type: <bool>
            Y:  (m x 1) Measurement value/vector
                type: <int>/<float> or <np.matrix>/<np.array>/<np.ndarray>
            H:  (m x n) Observation matrix 
                type: <int>/<float> or <np.matrix>/<np.array>/<np.ndarray>
        Output(s):
            x:  (n x 1) State(s) estimate(s)
                type: <np.matrix>
        '''
        if mc:
            for value in kwargs.values(): kwargs = value

        #Declare arguements
        Y = kwargs['Y']
        H = kwargs['H']

        #Assert valid input(s) type
        assert type(Y) == int or type(Y) == float or type(Y) == np.matrix or type(Y) == np.array or type(Y) == np.ndarray,\
            f"Input <Y> has invalid type. Expected <int>/<float> or <np.matrix>/<np.array>/<np.ndarray> but recieved <{type(Y)}>" 
        assert type(H) == int or type(H) == float or type(H) == np.matrix or type(H) == np.array or type(H) == np.ndarray,\
            f"Input <H> has invalid type. Expected <int>/<float> or <np.matrix>/<np.array>/<np.ndarray> but recieved <{type(H)}>"
        
        #Convert to matrix domain
        if (type(Y) != np.matrix):
            Y = np.asmatrix(Y)
        if (type(H) != np.matrix):
            H = np.asmatrix(H)

        #Assert valid input(s) type
        assert Y.shape[0] == self.m and Y.shape[1] == 1,\
            f"Input <Y> has invalid dimensions. Expected ({self.m},{1}) but recieved ({Y.shape[0]},{Y.shape[1]})"
        assert H.shape[0] == self.m and H.shape[1] == self.n,\
            f"Input <H> has invalid dimensions. Expected ({self.m},{self.n}) but recieved ({H.shape[0]},{H.shape[1]})"

        #Perform least-squares
        x = np.linalg.inv(H.T * H) * H.T * Y
        
        return x

    #----- Recursive Least Squares -----#
    def rls(self, mc:bool=False, **kwargs):
        '''
        Descrption: 
            Recursive least-squares estimation. 
            Algorithm designed to be embedded in an external iterative loop.
        Input(s):
            mc: Monte Carlo configuration
                type: <bool>
            Y:  (m x 1) Measurement value/vector
                type: <int>/<float> or <np.matrix>/<np.array>/<np.ndarray>
            H:  (m x n) Observation matrix 
                type: <int>/<float> or <np.matrix>/<np.array>/<np.ndarray>
            R:  (m x m) Measurement covariance matrix
                type: <int>/<float> or <np.matrix>/<np.array>/<np.ndarray>
            P:  (n x n) Estimate covariance matrix
                type: <int>/<float> or <np.matrix>/<np.array>/<np.ndarray>
            x:  (n x 1) State(s) estimate(s)
                type: <int>/<float> or <np.matrix>/<np.array>/<np.ndarray>
        Output(s):
            x:  (n x 1) State(s) estimate(s)
                type: <np.matrix>
            P:  (n x n) Estimate covariance matrix
                type: <np.matrix>
            L:  (n x m) Gain matrix
                type: <np.matrix>
            innov:  (m x 1) Measurement innovation
                    type: <np.matrix>
        '''

        if mc:
            for value in kwargs.values(): kwargs = value

        #Declare arugements
        Y = kwargs['Y']
        H = kwargs['H']
        R = kwargs['R']
        P = kwargs['P']
        x = kwargs['x']

        #Assert valid input(s) type
        assert type(Y) == int or type(Y) == np.int32 or type(Y) == float or type(Y) == np.matrix or type(Y) == np.array or type(Y) == np.ndarray,\
            f"Input <Y> has invalid type. Expected <int>/<float> or <np.matrix>/<np.array>/<np.ndarray> but recieved <{type(Y)}>"        
        assert type(H) == int or type(H) == float or type(H) == np.matrix or type(H) == np.array or type(H) == np.ndarray,\
            f"Input <H> has invalid type. Expected <int>/<float> or <np.matrix>/<np.array>/<np.ndarray> but recieved <{type(H)}>"        
        assert type(P) == int or type(P) == float or type(P) == np.matrix or type(P) == np.array or type(P) == np.ndarray,\
            f"Input <P> has invalid type. Expected <int>/<float> or <np.matrix>/<np.array>/<np.ndarray> but recieved <{type(P)}>"
        assert type(x) == int or type(x) == float or type(x) == np.matrix or type(x) == np.array or type(x) == np.ndarray,\
            f"Input <x> has invalid type. Expected <int>/<float> or <np.matrix>/<np.array>/<np.ndarray> but recieved <{type(x)}>"

        #Convert to matrix domain
        if (type(Y) != np.matrix):
            Y = np.asmatrix(Y)
        if (type(H) != np.matrix):
            H = np.asmatrix(H)
        if (type(P) != np.matrix):
            P = np.asmatrix(P)
        if (type(x) != np.matrix):
            x = np.asmatrix(x)

        #Assert valid input(s) shape
        assert Y.shape[0] == self.m and Y.shape[1] == 1,\
            f"Input <Y> has invalid dimensions. Expected ({self.m},{1}) but recieved ({Y.shape[0]},{Y.shape[1]})"
        assert H.shape[0] == self.m and H.shape[1] == self.n,\
            f"Input <H> has invalid dimensions. Expected ({self.m},{self.n}) but recieved ({H.shape[0]},{H.shape[1]})" 
        assert P.shape[0] == self.n and P.shape[1] == self.n,\
            f"Input <P> has invalid dimensions. Expected ({self.n},{self.n}) but recieved ({P.shape[0]},{P.shape[1]})"
        assert x.shape[0] == self.n and x.shape[1] == 1,\
            f"Input <x> has invalid dimensions. Expection ({self.n},{1}) but revieved ({x.shape[0]},{x.shape[1]})"

        #Theoretical specs
        dop = np.linalg.inv(H.T * H)                                            #Dilution of precision
        P_pred = np.sqrt(R * dop)                                               #Predicted state covariance

        #Perform recursive least squares
        L = P * H.T * np.linalg.inv(H * P * H.T + R*np.identity(self.m))        #Gain update
        innov = (Y - H*x)                                                       #Measurement innovation
        x = x + L*(innov)                                                       #State update
        P = (np.identity(self.n) - L*H)*P                                       #State covariance update

        return x, P, L, innov, dop, P_pred 