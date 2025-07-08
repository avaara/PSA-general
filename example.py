import qutip as Q
import numpy as np
import matplotlib.pyplot as plt
import itertools
import me_func_defs as mf

### Here we show as an example how to use the general partial secular
### approximation program to compute results of the paper Cattaneo et
### al. New J. Phys 21 (2019) 113045.



#Own definition of the spectral density function
def J(omega, khi, omega_c):
    return khi * omega/(1 + omega**2/omega_c**2)


def compute(params, which_baths):
    #----------System parameters for the qubit-qubit system-------------
    omega_1 = params[0]
    omega_2 = params[1]
    lambd = params[2]
    #-------------------------------------------------------------------

    #------------------Bath parameters------------------
    g_x_l1 = 1; g_z_l1 = 1
    g_x_l2 = 1; g_z_l2 = 1
    g_x_c1 = 1; g_x_c2 = 1
    g_z_c1 = 0; g_z_c2 = 0

    T_l1 = 1; T_l2 = 10; T_c = 1

    alpha_l1 = 0.01; alpha_l2 = 0.01; alpha_c = 0.01
    khi_l1 = 1; khi_l2 = 1; khi_c = 1
    #---------------------------------------------------

    #--------------------Building needed operators----------------------
    sigma_z = -Q.sigmaz()
    sigma_z1 = Q.tensor(sigma_z, Q.identity(2))
    sigma_z2 = Q.tensor(Q.identity(2), sigma_z)
    sigma_x1 = Q.tensor(Q.sigmax(), Q.identity(2))
    sigma_x2 = Q.tensor(Q.identity(2), Q.sigmax())
    #-------------------------------------------------------------------

    #----------Defining the system Hamiltonian-----------------
    H_S = (1/2*omega_1*sigma_z1 + 1/2*omega_2*sigma_z2
           + lambd*sigma_x1*sigma_x2)
    H_S_loc = 1/2*omega_1*sigma_z1 + 1/2*omega_2*sigma_z2
    #----------------------------------------------------------

    #-------------------Defining the baths---------------------
    bath_l1 = {"cpl_ops":[g_x_l1*sigma_x1 + g_z_l1*sigma_z1],
               "params":[T_l1, alpha_l1, khi_l1],
               "spectral_dens_func":J}
    bath_l2 = {"cpl_ops":[g_x_l2*sigma_x2 + g_z_l2*sigma_z2],
               "params":[T_l2, alpha_l2, khi_l2],
               "spectral_dens_func":J}
    bath_c = {"cpl_ops":[g_x_c1*sigma_x1 + g_x_c2*sigma_x2
                         + g_z_c1*sigma_z1 + g_z_c2*sigma_z2],
              "params":[T_c, alpha_c, khi_c],
              "spectral_dens_func":J}
    
    if(which_baths == "locals"):
        baths = [bath_l1, bath_l2]
    elif(which_baths == "common"):
        baths = [bath_c]
    elif(which_baths == "all"):
        baths = [bath_c, bath_l1, bath_l2]
    #----------------------------------------------------------

    #----------------Computing the Liouvillian-------------------
    psa_cut = params[3]
    L, u, disslist = mf.Liouvillian_general(H_S,
                                            baths,
                                            unified_eq=False,
                                            cluster_width=0.1,
                                            plot_clusters=False,
                                            psa_cut=psa_cut,
                                            omega_c=20,
                                            print_info=False)
    #------------------------------------------------------------

    #---------------Defining the initial state---------------
    q1_0 = 1/np.sqrt(2)*(Q.basis(2, 0) + Q.basis(2,1))
    q2_0 = 1/np.sqrt(2)*(Q.basis(2, 0) + Q.basis(2,1))

    psi_0 = Q.tensor(q1_0, q2_0)
    rho_0 = psi_0*psi_0.dag()
    #--------------------------------------------------------

    #----------------Computing the time evolution------------------
    t = np.linspace(0, 5000, num=1000)
    proj_exc = Q.tensor(Q.ket('e')*Q.bra('e'), Q.identity(2))
    res = mf.evolution([proj_exc], rho_0, t, L)[0]
    #--------------------------------------------------------------

    return res

#For Fig. 2a use these parameters and common bath
# omega1 = 1; omega2 = 0.99; lambd = 0; which_baths="common"

#For Fig. 3a use these parameters and local baths
# omega1 = 1; omega2 = 1; lambd = 1e-4; which_baths="locals"

#For Fig. 5c use these parameters and common bath
omega1 = 1; omega2 = 0.99; lambd = 1; which_baths="common"

#Partial secular approx has psa_cut>0 and full secular has psa_cut=0
psa_cut = 0

params = [omega1, omega2, lambd, psa_cut]
result = compute(params, which_baths=which_baths)
plt.plot(t, result)
plt.show(block=False)

