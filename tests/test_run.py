#%%
import numpy as np
import matplotlib.pyplot as plt

from filter_tools.estimators import Estimators
from python_utilities.parsers_class import Parsers
#from monte_carlo_test import monteCarlo

if __name__ == '__main__':

    # Example problem (source: Optimal Homework 2)
    a = 1.3
    b = 2
    w = 2 * np.pi
    t = np.arange(0,10,.01)
    t = t.reshape(-1,1)
    r = 100 * np.sin(w*t)
    R = np.deg2rad(0.3)**2
    n = np.sqrt(R)*np.random.randn(len(t))
    n = n.reshape(-1,1)
    g = a * r + b + n

    #Plotting
    fig, axs = plt.subplots(2)
    axs[0].plot(t, r)
    axs[0].set_title('Raw sine')
    axs[0].grid()
    axs[1].plot(t, g)
    axs[1].set_title('Gyro measurements')
    axs[1].grid()
    fig.tight_layout()
    plt.show()

    #-----LS Test-----
    #LS metrics
    Y_ls = g
    H_ls = np.hstack((r, np.ones_like(r)))

    #Run LS example
    print('\nRunning LS')

    ls_instance = Estimators(n=2, m=len(Y_ls))
    x_ls = ls_instance.ls(Y=Y_ls, H=H_ls)

    print('LS complete')

    #-----RLS Test-----
    #Parse config file
    rls_config_file = 'config/rls_config.yaml'
    rls_config = Parsers().yamlParser(yaml_file=rls_config_file)
    
    #Estimator metrics
    total_samples = len(t)
    rls_samples = 10
    rls_length = int(total_samples/rls_samples)

    #Initialize
    # x_rls = np.matrix(np.zeros((2,int(total_samples/rls_samples))))
    # x_rls[:,0] = np.matrix(rls_config["x_init"])
    # P_init = np.matrix(rls_config["P_init"])
    # P = P_init

    #Run RLS example
    print('\nRLS Running ...')

    rls_instance = Estimators(n=2, m=rls_samples)

    #Set monte carlo specs
    mc_length = 50
    x1_mc = np.matrix(np.zeros((mc_length, rls_length)))    #Initialize monte carlo matrices
    x2_mc = np.matrix(np.zeros((mc_length, rls_length)))
    x1_err_mc = np.matrix(np.zeros((mc_length, rls_length)))
    x2_err_mc = np.matrix(np.zeros((mc_length, rls_length)))
    for k in range(mc_length):
    
        n = np.sqrt(R)*np.random.randn(len(t))  #Update noise
        n = n.reshape(-1,1)
        g = a * r + b + n                       #Update measurement

        #Initialize
        x_rls = list()
        P_rls = list()
        L_rls = list()
        innov_rls = list()
        dop_rls = list()
        P_pred_rls = list()

        x = np.matrix(rls_config["x_init"])
        P = np.matrix(rls_config["P_init"])
        x_rls.append(x)
        P_rls.append(P)
        
        for i in range(int(total_samples / rls_samples)-1):

            #Generate measurement
            Y_rls = g[(i*rls_samples):((i+1)*rls_samples)]
            H_rls = np.hstack((r[(i*rls_samples):((i+1)*rls_samples)], np.ones(rls_samples).reshape(-1,1)))

            #RLS
            x, P, L, innov, dop, P_pred = rls_instance.rls(Y=Y_rls, H=H_rls, R=R, P=P, x=x)

            #Append outputs
            x_rls.append(x)
            P_rls.append(P)
            L_rls.append(L)
            innov_rls.append(innov)
            dop_rls.append(dop)
            P_pred_rls.append(P_pred)

        #Covert output to nd.array
        x_rls = np.asarray(x_rls)
        P_rls = np.asarray(P_rls)
        L_rls = np.asarray(L_rls)
        innov_rls = np.asarray(innov)
        dop_rls = np.asarray(dop_rls)
        P_pred_rls = np.asarray(P_pred)

        #Append monte carlo matrices
        x1_mc[k,:] = x_rls[:,0].T
        x2_mc[k,:] = x_rls[:,1].T

        x1_err_mc[k,:] = x_rls[:,0].T - a
        x2_err_mc[k,:] = x_rls[:,1].T - b

    x1_mean = np.ravel(np.average(x1_mc,0))
    x1_std = np.ravel(np.std(x1_mc,0))
    x1_err_mean = np.ravel(np.average(x1_err_mc,0))
    x1_err_std = np.ravel(np.std(x1_err_mc,0))

    x2_mean = np.ravel(np.average(x2_mc,0))
    x2_std = np.ravel(np.std(x2_mc,0))
    x2_err_mean = np.ravel(np.average(x2_err_mc,0))
    x2_err_std = np.ravel(np.std(x2_err_mc,0))

    #Theoretical std
    x1_std_theo = np.sqrt(P_rls[:,0,0])
    x2_std_theo = np.sqrt(P_rls[:,1,1])

    print('RLS Completed\n')

    #Plot MC a
    fig, axs = plt.subplots(2)
    axs[0].plot(x1_mc[:,1:].T, color=(0.7,0.7,0.7))
    axs[0].plot(x1_mean[1:] + 3*x1_std[1:], 'r', label='Experimental')
    axs[0].plot(x1_mean[1:] - 3*x1_std[1:], 'r')
    axs[0].plot(x1_mean[1:] + 3*x1_std_theo[1:], '--k', label='Theoretical')
    axs[0].plot(x1_mean[1:] - 3*x1_std_theo[1:], '--k')
    axs[0].set_title('a Estimates')
    axs[0].set_xlim([0,50])
    axs[0].legend()
    axs[0].grid()
    axs[1].plot(x1_err_mc[:,1:].T, color=(0.7,0.7,0.7))
    axs[1].plot(x1_err_mean[1:] + 3*x1_err_std[1:], 'r', label='Experimental')
    axs[1].plot(x1_err_mean[1:] - 3*x1_err_std[1:], 'r')
    axs[1].plot(x1_err_mean[1:] + 3*x1_std_theo[1:], '--k', label='Theoretical')
    axs[1].plot(x1_err_mean[1:] - 3*x1_std_theo[1:], '--k')
    axs[1].set_title('a Estimate Errors')
    axs[1].set_xlim([0,50])
    axs[1].set_xlabel('RLS Iterations')
    axs[1].grid()
    
    fig.tight_layout()
    plt.show()

    #plot MC b
    fig, axs = plt.subplots(2)
    axs[0].plot(x2_mc[:,1:].T, color=(0.7,0.7,0.7))
    axs[0].plot(x2_mean[1:] + 3*x2_std[1:], 'r', label='Experimental')
    axs[0].plot(x2_mean[1:] - 3*x2_std[1:], 'r')
    axs[0].plot(x2_mean[1:] + 3*x2_std_theo[1:], '--k', label='Theoretical')
    axs[0].plot(x2_mean[1:] - 3*x2_std_theo[1:], '--k')
    axs[0].set_title('b Estimates')
    axs[0].set_xlim([0,50])
    axs[0].legend()
    axs[0].grid()
    axs[1].plot(x2_err_mc[:,1:].T, color=(0.7,0.7,0.7))
    axs[1].plot(x2_err_mean[1:] + 3*x2_err_std[1:], 'r', label='Experimental')
    axs[1].plot(x2_err_mean[1:] - 3*x2_err_std[1:], 'r')
    axs[1].plot(x2_err_mean[1:] + 3*x2_std_theo[1:], '--k', label='Theoretical')
    axs[1].plot(x2_err_mean[1:] - 3*x2_std_theo[1:], '--k')
    axs[1].set_title('b Estimate Errors')
    axs[1].set_xlim([0,50])
    axs[1].set_xlabel('RLS Iterations')
    axs[1].grid()
    fig.tight_layout()
    plt.show()
