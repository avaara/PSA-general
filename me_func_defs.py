import qutip as Q
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.constants import h, hbar, Boltzmann
from alive_progress import alive_bar
import itertools
import os.path
import time as ttime

Q.settings.auto_tidyup = False

np.set_printoptions(linewidth=500)


def J(omega_norm, khi, omega_c_norm=100): 
    """
    Defines the (normalized) spectral density of the bath.

    Parameters
    ----------
    omega_norm : float
        Normalized frequency for which to compute the spectral density
    khi : float
        A parameter describing the strength of coupling to bath
    omega_c_norm : float (default 100)
        A cutoff frequency
    
    Returns
    -------
     : float
        The computed spectral density

    """
    # khi = 1 #3/160
    # The normalized cut-off frequency
    return khi * omega_norm/(np.pi*(1 + omega_norm**2/omega_c_norm**2))

def S(omega_norm, khi, temperature, J_func, omega_c=100):
    """
    Defines the function S(omega).

    The S(omega) function is needed to compute the Lamb-shift and the
    emission/absorbtion coefficients when the full secular approximation
    is not applied.

    Parameters
    ----------
    omega_norm : float
        Normalized frequency, used to call the function J_func
    khi : float
        Parameter, used to call the function J_func
    temperature : float
        Temperature of the bath, used to call function BE_distribution
    J_func : function
        The spectral density function of the bath
    omega_c_norm : float (default 100)
        The cutoff frequency for the spectral density function J_func

    Returns
    -------
     : float
        The computed value for S

    Notes
    -----
    The value for S is computed via numerical integration with a Cauchy weight.
    The used weight comes from the definition of the integrand.

    Ideally, the integration limits should be from 0 to infinity (-infinity to 0).
    However, that causes division by zero errors and np.infinity cannot be used with
    Cauchy weight. Therefore the limits are set to values which are small and large
    enough such that the integral value does not change when the limits are changed. 
    This was done by comparing the numerical results between this implementation and 
    Mathematica, where 0 and infinity were valid limits. The results agree to 6 decimal 
    places.
    
    """

    # print("\nComputing S with frequency {}".format(omega_norm))

    #Computes S(omega) for positive frequencies
    if(omega_norm > 0):
        I1 = integrate.quad(lambda x: -J_func(x*omega_norm, khi, omega_c)*(1 + BE_distribution(x*omega_norm, temperature)),
                            1e-6,
                            1e9,
                            weight='cauchy',
                            wvar=1,
                            epsabs=1e-10)
        I2 = integrate.quad(lambda x: J_func(x*omega_norm, khi, omega_c)*BE_distribution(x*omega_norm, temperature),
                            1e-6,
                            1e9,
                            weight='cauchy',
                            wvar=-1,
                            epsabs=1e-10)
        # print("   Result: {}".format(I1[0] + I2[0]))
        return I1[0] + I2[0]

    #Computes S(omega) for negative frequencies
    elif(omega_norm < 0):
        I1 = integrate.quad(lambda x: -J_func(x*omega_norm, khi, omega_c)*(1 + BE_distribution(x*omega_norm, temperature)),
                            -1e9,
                            -1e-6,
                            weight='cauchy',
                            wvar=1,
                            epsabs=1e-10)
        I2 = integrate.quad(lambda x: J_func(x*omega_norm, khi, omega_c)*BE_distribution(x*omega_norm, temperature),
                            -1e9,
                            -1e-6,
                            weight='cauchy',
                            wvar=-1,
                            epsabs=1e-10)
        # print("   Result: {}".format(-I1[0] - I2[0]))
        return -I1[0] - I2[0]

    #This is the case omega = 0.0
    else:
        I1 = integrate.quad(lambda x: -J_func(x*omega_norm, khi, omega_c),
                            1e-6,
                            1e9,
                            weight='cauchy',
                            wvar=0,
                            epsabs=1e-10)
        return I1[0]

def gamma(omega1, omega2, khi, temperature, J_func, omega_c=100):
    """
    Defines the emission and absorbtion coefficients.

    Used in the dissipators of master equations. In the definition
    the secular approximation has not been applied, so the returned
    gamma depends on two frequencies of the underlying jump operators.

    Parameters
    ----------
    omega1 : float
        First frequency of a jump operator
    omega2 : float
        Second frequency of another jump operator
    khi : float
        Parameter used for calling the function J
    temperature : 
        Temperature of the bath, used to call function BE_distribution

    Returns
    -------
     : complex
        The computed emission/absorbtion coefficient

    Notes
    -----
    The returned value can be complex if the secular approximation is
    not applied, setting omega1 == omega2. If the secular approximation
    is done, then the returned value is real.

    """

    print("\nComputing gamma with frequencies {}, {}".format(omega1, omega2))
    
    if(omega1 > 0):
        real1 = np.pi * J_func(omega1, khi, omega_c) * (1 + BE_distribution(omega1, temperature))
        # print("Re:{}".format(real1))
    elif(omega1 < 0):
        real1 = np.pi * J_func(-omega1, khi, omega_c) * BE_distribution(-omega1, temperature)
        # print("Re:{}".format(real1))
    else:
        # real1 = 0
        # Ad hoc limit of omega1 going to zero
        real1 = np.pi * J_func(1e-8, khi, omega_c) * (2*BE_distribution(1e-8, temperature) + 1)

    if(omega2 > 0):
        real2 = np.pi * J_func(omega2, khi, omega_c) * (1 + BE_distribution(omega2, temperature))
    elif(omega2 < 0):
        real2 = np.pi * J_func(-omega2, khi, omega_c) * BE_distribution(-omega2, temperature)
    else:
        # real2 = 0
        # Ad hoc limit of omega2 going to zero
        real2 = np.pi * J_func(1e-8, khi, omega_c) * (2*BE_distribution(1e-8, temperature) + 1)

    imag1 = S(omega1, khi, temperature, J_func, omega_c)
    imag2 = S(omega2, khi, temperature, J_func, omega_c)

    # print("Re:{}, {}".format(real1, real2))
    # print("Im:{}, {}".format(imag1, imag2))
    
    return real1 + real2 + 1j*imag1 - 1j*imag2

def BE_distribution(omega_norm, temperature, natural_units=True):
    """
    Defines the standard Bose-Einstein distribution.

    Parameters
    ----------
    omega_norm : float
        Normalized frequency of the boson
    temperature : float
        Temperature of the bath
    natural_units : bool (default True)
        Whether to use natural units where hbar = kB = 1 or SI units

    Returns
    -------
    BE : float
        The average number of bosons to be found with given frequency and
        temperature.

    Notes
    -----
    The factor C gives the energyscale in question depending on the frequency.

    If the frequencies grow much higher than temperature, such that C >> 1
    the exponential is very small and returns an OverflowError. This is caught 
    and zero is returned instead.
    
    """

    # print("Computing BE with frequency {}".format(omega_norm))

    if(natural_units):
        C = 1/temperature #Here hbar = kB = 1
    else:
        C = hbar*omega_scale/(Boltzmann*temperature)
        
    try:
        if(abs(omega_norm)<1e-6):
            # Taylor expansion for very small omega
            BE = 1/(C*omega_norm) #+ 0.5*(1/C*omega_norm)**2
        else:
            BE = 1/(np.e**(omega_norm*C) - 1)
        #print("Result is {}".format(BE))
        return BE
    except OverflowError: #catches error due to too small number
        #print("Returning n=0 for BE-distribution. Frequency is {}".format(omega_norm))
        return 0

def op_to_vec(matrix):
    """
    Vectorization of an operator by stacking the rows.

    QuTip uses column stacking when it works in the Liouville space. My
    calculation are done using row-stacking so this function replaces the
    QuTip column-stacking function with row-stacking equivalent.

    Parameters
    ----------
    matrix : Qobj
        The Qobj operator (in matrix form) which needs to be vectorized.

    Returns
    -------
    vec : Qobj
        The vectorized form of the original operator.
    
    """
    
    #--------------------------------------------------------
    #Building the right dimensions for the vectorized operator
    #Complies with the definition from QuTiP source code
    dims = matrix.dims
    dims = [dims]
    dims.append([1])
    #---------------------------------------------------------
    
    data = []
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            data.append([matrix[row][0][col]])

    vec = Q.Qobj(data, dims)
    return vec

def vec_to_op(vec):
    """
    Inverse operation to vectorization function op_to_vec.

    QuTip uses column stacking when it works in the Liouville space. My
    calculation are done using row-stacking so this function replaces the
    QuTip column-stacking function with row-stacking equivalent.

    Parameters
    ----------
    vec : Qobj
        Vectorized (ket) form of an operator.

    Returns
    -------
    op : Qobj
        Operator obtained from vec by doing the inverse operation to row
        stacking. An operator in matrix form.
    
    """
    
    #--------------------------------------------------------
    #Initializing an empty matrix 'op' to be filled with values
    dims = vec.dims[0]
    #print("dims: {}".format(dims))
    #D = np.prod(dims[0]) #Dimension of matrix is DxD
    D = int(np.sqrt(vec.shape[0]))
    #print("D: {}".format(D))
    op = [[0 for x in range(D)] for x in range(D)]
    #--------------------------------------------------------
    
    for row in range(D):
        for col in range(D):
            op[row][col] = vec[col + row*D][0][0]

    op = Q.Qobj(op, dims)
    return op

def do_secular_approx(omega1,
                      omega2,
                      timescale,
                      prints=False,
                      psa_cut=1e4,
                      tol=1e-10):
    """
    Determines whether or not the secular approximation can be safely performed.

    Parameters
    ----------
    omega1 : float
        Frequency related to one jump operator.
    omega2 : float
        Frequency related to the second jump operator.
    timescale : float
        Timescale of the system time evolution.
    prints : bool (default False)
        Whether or not to print helping text to see what the code is doing.
    psa_cut : float (default 1e4)
        Cutoff parameter for the partial secular approximation
    tol : float (default 1e-10)
        Tolerance within which a number is treated as purely zero

    Returns
    -------
    dropTerm : bool
        Whether or not the secular approximation can be applied to this term where the pair of
        jump operators are present.

    Notes
    -----
    The secular approximation can be done if the timescale related to the jump operator
    frequency difference is much smaller than the timescale of the system time evolution.
    Here 'much smaller than' means that tau_freqdiff*psa_cut < tau_system. The larger the cutoff,
    the smaller the frequency difference needs to be in order for the term to be neglected.
    Therefore infinite psa_cut means that all terms are taken into account and zero psa_cut
    refers to full secular approximation.

    In the case of psa_cut = 0, the freq_diff_timescale*psa_cut = nan when omega1=omega2.
    Therefore the algorithm is able to perform FSA, because 0*np.infinity = nan and
    nan < timescale is False, which sets dropTerm = False.

    If the condition tau_freqdiff*psa_cut < tau_system is True, the secular approximation
    can be performed and the term discarded.

    """

    try:        
        delta = abs(omega1 - omega2)
        if(delta < tol):
            delta = 0
        freq_diff_timescale = 1/delta
    except ZeroDivisionError: #Sets timescale to infinity if omega1 == omega2
        freq_diff_timescale = np.inf
        
    if(freq_diff_timescale*psa_cut < timescale):
        dropTerm = True
        if(prints):
            msg = "  psa_cut={psa}, {psa}*{fdt:.2f}={prod:.2f} < {ts:.2f} -> DISCARD"
            print(msg.format(psa=psa_cut, fdt=freq_diff_timescale,
                             prod=psa_cut*freq_diff_timescale, ts=timescale))
    else:
        dropTerm = False
        if(prints):
            msg = "  psa_cut={psa}, {psa}*{fdt:.2f}={prod:.2f} > {ts:.2f} -> KEEP"
            print(msg.format(psa=psa_cut, fdt=freq_diff_timescale,
                             prod=psa_cut*freq_diff_timescale, ts=timescale))

    return dropTerm

def Scoeff(omega, omega2, khi, temperature, J_func, omega_c=100):
    """
    Computes the coefficient S(omega, omega') which is needed for the Lamb shift.

    Parameters
    ----------
    omega : float
        Frequency of the jump operator, used for calling function S.
    omega2 : float
        Second frequency of the jump operators, used for calling function S.
    khi : float
        Parameter needed for spectral density function J.
    temperature : float
        Temperature of the bath, needed for computing BE_distribution.

    Returns
    -------
     : complex
        Scoefficient for computing the Lamb-shift

    Notes
    -----
    The coefficient has a real and imaginary part, where the real part is computed using
    the S(omega) function defined before as a principal value integral. The imaginary part
    vanishes in full secular approximation, when omega = omega'.

    """

    #The sign of omega is taken care of in the function computing S(omega)
    realpart = 1/2*(S(omega, khi, temperature, J_func, omega_c)
                    + S(omega2, khi, temperature, J_func, omega_c))

    #The imaginary part is different if the frequencies have different signs
    if(omega > 0): #Emission
        imag1 = np.pi/2 * J_func(omega, khi, omega_c) * (1 + BE_distribution(omega, temperature))
    if(omega < 0): #Absorbtion
        imag1 = np.pi/2 * J_func(-omega, khi, omega_c) * BE_distribution(-omega, temperature)
    else:
        # imag1 = 0
        # Ad hoc limit of omega going to zero
        imag1 = np.pi/2 * J_func(1e-8, khi, omega_c) * (2*BE_distribution(1e-8, temperature) + 1)
        

    if(omega2 > 0): #Emission
        imag2 = np.pi/2 * J_func(omega2, khi, omega_c) * (1 + BE_distribution(omega2, temperature))
    if(omega2 < 0): #Absorbtion
        imag2 = np.pi/2 * J_func(-omega2, khi, omega_c) * BE_distribution(-omega2, temperature)
    else:
        # imag2 = 0
        # Ad hoc limit of omega2 going to zero
        imag2 = np.pi/2 * J_func(1e-8, khi, omega_c) * (2*BE_distribution(1e-8, temperature) + 1)

    imagpart = imag1 - imag2

    return complex(realpart, -imagpart)

def jump_ops_general(Hamiltonian, coupling_ops, tol=1e-10):
    """
    Computes the jump operators of the given coupling operators.

    Parameters
    ----------
    Hamiltonian : Qobj
        The system Hamiltonian with which the jump operators are computed.
    coupling_ops : list of Qobjs
        The coupling operators of the bath to the system Hamiltonian.
    tol : float (default 1e-10)
        The tolerance within which two float are treated as equal.

    Returns
    -------
    jump_ops : a list of lists
        Each separate list corresponds to the jump operators related to the
        specific coupling operator.

    Notes
    -----
    If the coupling operator to the bath is, for example, a.dag() - a, then
    the final jump_ops has only one list, namely all A(\omega) that correspond
    to jumps computed with the coupling operator a.dag() - a. However, one can also
    split this to two parts, namely [a.dag(), -a] and then the returned jump_ops is
    a list of two lists, first of which corresponds to jump operators of a.dag()
    and the second to jump operators of -a.
    
    """

    eigenvals_vecs = Hamiltonian.eigenstates()

    #Making a list of all eigenvalue, -vector pairs
    eigenspace = np.array(list(zip(eigenvals_vecs[0], eigenvals_vecs[1])), dtype=tuple)

    #Pairs each of the eval, evec pairs to another pair. These are used in computing
    #the jump operator from state to state.
    pairs =  np.array(list(itertools.permutations(eigenspace, r=2)), dtype=tuple)

    jump_ops = [[] for _ in range(len(coupling_ops))]
    for pair in pairs:
        eval1 = pair[0][0]
        evec1 = pair[0][1]
        eval2 = pair[1][0]
        evec2 = pair[1][1]

        energy_jump = eval2 - eval1

        #For the general situation if there are multiple coupling operators
        for i, op in enumerate(coupling_ops):
            #This is <psi_1|O|psi_2>
            coeff = op.matrix_element(evec1.dag(), evec2)
            #And the jump operator is <psi_1|O|psi_2> |psi_1><psi_2|
            jump = coeff * evec1*evec2.dag()

            #Taking into account only the ones which are non-zero with given tolerance
            if(np.abs(coeff) > tol):
                jump_ops[i].append([i+1, (eval2, eval1), energy_jump, jump])
                # print("Jump energy: {}".format(energy_jump))
                # print("Coeff: {}".format(coeff))
                # print(jump)

    return jump_ops

def Lamb_shift_general(jump_operators,
                       parameters,
                       J_func,
                       psa_cut=1e4,
                       tol=1e-10,
                       omega_c=100):
    """
    Computes the Lamb-shift due to the environmental interaction to the 
    system Hamiltonian. 

    The Lamb-shift is to renormalizes the system's eigenenergies.

    Parameters
    ----------
    jump_operators : a list of lists
        Each separate list refers to the jump operators corresponding to a specific
        coupling operator to the given bath.
    parameters : list of floats
        Parameters of the bath and the related coupling to the system.
    psa_cut : float (default 1e4)
        The cut off parameter for performing the partial secular approximation.
    tol : float (default 1e-10)
        The tolerance within which two float are treated as equal.

    Returns
    -------
    H_LS : Qobj
        The Lamb shift Hamiltonian coming from the given bath.
    
    """
    T = parameters[0]
    alpha = parameters[1]
    chi = parameters[2]

    #Setting the timescale of dynamics for the secular approximation
    tscale = 1/chi * 1/alpha**2

    #Initializing the Lamb-shift object
    H_LS = Q.Qobj()

    #Goes through jump operators coming from the different coupling operators (sum over beta)
    for j_op_list in jump_operators:
        #Makes jump operator pairs
        jump_pairs = np.array(list(itertools.combinations_with_replacement(j_op_list, 2)))

        #Going through the jump operator combinations
        for pair in jump_pairs:
            jump1 = pair[0]
            jump2 = pair[1]

            beta1 = jump1[0]
            omega1 = jump1[2]
            op1 = jump1[3]

            beta2 = jump2[0]
            omega2 = jump2[2]
            op2 = jump2[3]

            #Checking if the secular approximation is valid for this pair with given frequencies
            #If yes, then discards the pair and continues to the next one
            if(do_secular_approx(omega1, omega2, tscale, psa_cut=psa_cut, tol=tol)):
                continue

            Sfactor = Scoeff(omega1, omega2, chi, T, J_func, omega_c)

            #If jump operators are the same then just add them once
            if(op1 == op2):
                H_LS += alpha**2 * Sfactor * op1.dag()*op2
                continue

            #If the operators are not the same and the secular approximation still is invalid,
            #add the operator pair and its conjugate to the Lamb-shift
            H_LS += alpha**2 * Sfactor * op1.dag()*op2
            H_LS += alpha**2 * np.conjugate(Sfactor) * op2.dag()*op1

    return H_LS
                
def dissipator_general(jump_operators,
                       parameters,
                       J_func,
                       psa_cut=1e4,
                       tol=1e-10,
                       omega_c=100,
                       print_info=False):
    """
    Computes the dissipator part of the master equation.

    Parameters
    ----------
    jump_operators : a list of lists
        Each separate list refers to the jump operators corresponding to a specific
        coupling operator to the given bath.
    parameters : list of floats
        Parameters of the bath and the related coupling to the system.
    psa_cut : float (default 1e4)
        The cut off parameter for performing the partial secular approximation.
    tol : float (default 1e-10)
        The tolerance within which two float are treated as equal.
    print_info : bool (default False)
        Whether to print extra information during running the code.

    Returns
    -------
    dissipator : Qobj
        The dissipator of the master equation corresponding to the given bath

    """
    
    T = parameters[0]
    alpha = parameters[1]
    chi = parameters[2]

    #Setting the timescale of dynamics for the secular approximation
    tscale = 1/chi * 1/alpha**2

    #Initializing the dissipator
    dissipator = Q.Qobj()

    #Goes through jump operators coming from the different coupling operators (sum over beta)
    discarded = 0
    for j_op_list in jump_operators:

        #Building the needed identity operator
        dim = j_op_list[0][3].shape[0]
        Id = Q.qeye(dim)
        Id = Q.Qobj(Id, dims=j_op_list[0][3].dims)
    
        #Makes jump operator pairs
        jump_pairs = list(itertools.combinations_with_replacement(j_op_list, 2))
        
        #Going through the jump operator combinations
        for pair in jump_pairs:
            jump1 = pair[0]
            jump2 = pair[1]

            beta1 = jump1[0]
            omega1 = jump1[2]
            op1 = jump1[3]

            beta2 = jump2[0]
            omega2 = jump2[2]
            op2 = jump2[3]

            #Checks whether the secular approximation should be performed
            #for the given jump operator pair
            if(do_secular_approx(omega1, omega2, tscale, prints=print_info, psa_cut=psa_cut, tol=tol)):
                if(print_info): print("    Jump operator pair was discarded.")
                discarded += 1
                continue

            y = alpha**2 * gamma(omega1, omega2, chi, T, J_func, omega_c) #This is the factor gamma in front of dissipator
            # print("y: {}, omega1: {}, omega2: {}".format(y, omega1, omega2))
            # print(pair)
            #If jump operators are the same, add them only once
            if(op1 == op2):
                this_diss = y * (Q.tensor(op1, op2.dag().trans())
                                 - 1/2*Q.tensor(op2.dag()*op1, Id)
                                 - 1/2*Q.tensor(Id, (op2.dag()*op1).trans()))
                dissipator += this_diss
                if(print_info): print("    Equal pair was added with gamma = {}".format(y))
                continue


            #If the jump operators are not the same, add to the dissipator
            #the combination and its conjugate
            this_diss1 = y * (Q.tensor(op1, op2.dag().trans())
                              - 1/2*Q.tensor(op2.dag()*op1, Id)
                              - 1/2*Q.tensor(Id, (op2.dag()*op1).trans()))
            this_diss2 = np.conjugate(y) * (Q.tensor(op2, op1.dag().trans())
                                            - 1/2*Q.tensor(op1.dag()*op2, Id)
                                            - 1/2*Q.tensor(Id, (op1.dag()*op2).trans()))
        
            dissipator += this_diss1 + this_diss2
            if(print_info): print("    Cross term was added with gamma = {}".format(y))

        
        print("...Out of {} jump operator pairs {} were discarded due to secular approximation"
              .format(len(jump_pairs), discarded))
    return dissipator

def dissipator_unified_eq(jump_operators,
                          clusters_and_avgs,
                          parameters,
                          J_func,
                          print_info=False):
    """
    Computes the dissipator of the unified master equation approach by performing
    the PSA between clusters of Bohr frequencies while the jump operators within
    the same cluster are summed together.

    Parameters
    ----------
    jump_operators : : a list of lists
        Each separate list refers to the jump operators corresponding to a specific
        coupling operator to the given bath.
    clusters_and_avgs : a list [clusters, averages]
        Clusters is list of lists where each cluster is a list containing the Bohr
        frequencies belonging to that cluster. Averages is a list containing the
        average frequency of each cluster.
    parameters : parameters : list of floats
        Parameters of the bath and the related coupling to the system.

    Returns
    -------
    diss : Qobj
        The dissipator of the master equation corresponding to the given bath.


    Notes
    -----
    The unified master equation is a ME of the Redfield form, where the PSA is
    performed by utilizing clustering of the Bohr frequencies of each jump operator.
    Namely, the PSA is performed such that jump operator pairs with Bohr frequencies
    belonging to different clusters are neglected. Within the same cluster, the cluster
    average \bar{\omega} is used for computing the coefficient \gamma, whereas the
    collective jump operator of the cluster is obtained by summing all the jump operators
    of that cluster together.

    In order to use this method, one should ensure that there is a clear clustering
    of Bohr frequencies, such that the spectrum is not flat.
    
    """

    clusters = clusters_and_avgs[0]
    clus_avgs = clusters_and_avgs[1]

    print("...Computing unified me with {} clusters".format(len(clus_avgs)))

    #Initializing array of clusters to search for index of specific Bohr freq
    clusters_arr = np.empty([len(clusters), len(max(clusters, key=lambda x: len(x)))])
    clusters_arr[:] = np.nan

    #Adding frequencies of each cluster to the 2D array. Each row is a cluster
    for i, clus in enumerate(clusters):
        clusters_arr[i][0:len(clus)] = clus

    new_jump_ops = [[] for l in jump_operators]
    
    #Goes through jump operators coming from the different coupling operators (sum over beta)
    for j, j_op_list in enumerate(jump_operators):
        clustered_jumps = [[] for arr in clusters]

        #Searching for the jump op corresponding to specific Bohr freq and adding the jump
        #in to list of clustered jump operators to its correct place. Not all clusters have
        #jump operators.
        for jump in j_op_list:
            Bohr_f = jump[2]
            #Searching the index of the correct Bohr freq in the clusters
            list_idx = tuple(np.where(np.round(clusters_arr, 8) == round(Bohr_f, 8))[0])[0]
            clustered_jumps[list_idx].append(jump)

        #print(clustered_jumps)
        #Computes the sum of jumps in the same cluster
        for i, jump_cluster in enumerate(clustered_jumps):
            tot_jump = Q.Qobj()
            for jump in jump_cluster:
                tot_jump += jump[3]

            avg_omega = clus_avgs[i]
            # print(i)
            # print(tot_jump)
            #Adding the summed up jump operator of the cluster to a list
            #that has the same form as the jump_operators list before. The
            #following discards the clusters with no jump operators.
            if(not (tot_jump.isbra == True and tot_jump.shape == (1,1))):
                new_jump_ops[j].append([j+1, np.nan, avg_omega, tot_jump])


    print("...of which {} clusters were assigned a single unified jump op"
          .format(sum(len(new_jump_ops[i]) for i in range(len(new_jump_ops)))))
    #The dissipator of the unified me can be calculated using the same method
    #as for the general case because only the jumps have been grouped together
    #differently. The full secular aprox is performed between the jumps
    #from different clusters.
    diss = dissipator_general(new_jump_ops, parameters, J_func, psa_cut=0, print_info=print_info)

    return diss

def freq_clustering(Bohr_fs, width, plot=False):
    """
    Finds the Bohr frequency clusters needed for the application of
    the unified master equation, which is another way of performing
    the PSA.

    Parameters
    ----------
    Bohr_fs : a list
        A list of Bohr frequencies i.e. the energy level differences of
        the system Hamiltonian eigenenergies.
    width : float
        Required width between two consecutive Bohr fequencies before
        the next cluster starts.
    plot : bool (default False)
        Whether to plot the Bohr frequencies and the resulting clusters.

    Notes
    -----
    The algorithm check only consecutive Bohr frequencies and if their difference
    is larger than the provided width, a new cluster begins. Therefore within each
    cluster, the frequencies are nearer to each other than between clusters. However
    for equally spaced energy spectrum this algorithm creates either N clusters (where
    N is the number of the distinct frequencies) or only one cluster. Therefore one
    should check the frequency spacing by plotting the clusters before using the
    unified master equation approach such that one ensures that the frequency spectrum
    has well defined gaps which can be clustered in a meaningful manner.
    
    """

    #Sorting the list of provided Bohr frewuencies
    Bohr_fs = np.sort(Bohr_fs)

    #Building the clusters
    clusters = [] #list for all clusters
    clus = []     #list for a single cluster
    for i in range(len(Bohr_fs) - 1):
        nn_diff = Bohr_fs[i+1] - Bohr_fs[i] #nearest neighbout difference between two freqs
        
        #When difference is smaller than width, \omega_i belongs to cluster
        if(nn_diff < width):                
            clus.append(Bohr_fs[i])
            continue

        #When difference to the next is larger, add current frequency to belonging cluster and start a new one
        else:
            clus.append(Bohr_fs[i])
            clusters.append(clus)
            clus = []
            continue

    #Clustering for the last frequency
    f_l = Bohr_fs[-1]
    nn_diff_l = Bohr_fs[-1] - Bohr_fs[-2]
    if(nn_diff_l < width):
        clus.append(f_l)
        clusters.append(clus)
    else:
        if(not clus==[]):
            clusters.append(clus)
        clusters.append([f_l])

    #Computing the cluster averages
    averages = [np.average(cluster) for cluster in clusters]

    #Plotting the frequencies and coloring each cluster
    if(plot):
        colors = ['#FFB300', '#803E75', '#FF6800', '#A6BDD7', '#C10020', '#CEA262', '#817066',
                  '#007D34', '#F6768E', '#00538A', '#FF7A5C', '#53377A', '#FF8E00', '#B32851',
                  '#F4C800', '#7F180D', '#93AA00', '#593315', '#F13A13', '#232C16']
        j=0
        for i, clus in enumerate(clusters):
            if(j<len(colors)-1):
                j+=1
            else:
                j=0
            plt.hlines(clus, xmin=0.02, xmax=1, color=colors[j])

        plt.ylabel("Jump frequency")
        plt.xticks([], [])
        plt.plot([0 for _ in averages], averages, 'ro')
        plt.show()

    
    return [clusters, averages]
                
def Liouvillian_general(Hamiltonian_sys,
                        baths,
                        local_Hamiltonian=None,
                        include_LS=True,
                        unified_eq=False,
                        cluster_width=0.1,
                        plot_clusters=False,
                        psa_cut=1e4,
                        tol=1e-10,
                        omega_c=100,
                        print_info=False):
    """
    Builds the Liouvillian superoperator for computing the open quantum system dynamics.

    Parameters
    ----------
    Hamiltonian_sys : Qobj
        System Hamiltonian.
    baths : list of disctionaries [bath1_dict, bath2_dict, ...]
        Defines the baths connected to the system Hamiltonian. Each disctionary is a separate bath.
        Within each bath dictionary the keys are: "cpl_ops", "params" and "spectral_dens_func".
        1. The key "cpl_ops" gives a list of coupling operators of the system that couple
           to the bath (the jump operators are built from these).
        2. The key "params" is a list of parameters related to the system-bath coupling and bath temperature.
        3. The key "spectral_dens_func" gives the bath spectral density function. Only the function name
           should be provided (i.e. not call the function with its arguments). The function must take three
           parameters omega (for frequency), chi (for scaling the strength) and omega_c (frequency cutoff)
           exactly in this order. If this key has no value assigned to it, or the value is 'None', the
           predefined function J is used instead.
    local_Hamiltonian : Qobj (default None)
        If this is provided, the jump operators are computed w.r.t. this Hamiltonian. This
        is meant for manually supplying the local form of the system Hamiltonian (i.e. where
        the inner system interactions are neglected).
    include_LS : bool (default True)
        Determines if the Lamb-shift is computed and taken into account. Can be used for testing
        the effect of Lamb-shift to the dynamics and for speeding up the code if set to False.
    unified_eq : bool (default False)
        Whether to apply the unified master equation computation to perform the PSA.
    cluster_width : float (default 0.1)
        A parameter needed for the Bohr frequency clustering, which is a central part of the
        unified master equation method.
    plot_clusters : bool (default False)
        Whether to plot the clusters the algorithm found with given cluster width.
    psa_cut : float (default 1e4)
        A number for specifying the cut-off for the partial secular approximation. Larger value
        takes more terms into account while psa_cut=0 means full secular approximation.
    tol : float (default 1e-10)
        Tolerance within which two numbers are treated as equal.
    print_info : bool (default False)
        Whether to print extra information during running the code.

    Returns
    -------
     : list
        First element in the list is the full Liouvillian superoperator matrix, the second
        is the unitary part and last element is a list of dissipators for each provided bath.

    Notes
    -----
    The dissipators of each bath are built separately
    and then added together to form the full dissipator. Also the unitary part
    is constructed on it's own and then added to dissipators to create the full
    Liouvillian.

    Whether or not the secular approximation can be applied to the jump operator
    pairs is determined by the relevant timescales present in the system.

    Whether or not the Liouvillian is global or local depends just on the jump
    operators provided as a function argument. The jump operators are computed
    outside of this function, so if the jump ops are calculated with local
    Hamiltonian (i.e. with inner system interactions neglected), then the Liouvillian
    is also local automatically.
    """

    if(local_Hamiltonian != None):
        do_local = True
    else:
        do_local = False
    
    H_LS_tot = 0*Hamiltonian_sys
    diss_tot = Q.Qobj()
    diss_list = []
    
    for i, bath in enumerate(baths):
        print("\nBath {}:".format(i+1))

        #----------Unpacking the bath dictionary----------
        cpl_ops = bath["cpl_ops"]
        params = bath["params"]

        #This checks whether the user has provided their own spectral density function
        try:
            J_func = bath["spectral_dens_func"]
            if(callable(J_func)):
                pass
            elif(J_func==None):
                J_func = J
            else:
                msg = "You need to provide a callable function for the spectral density, \
                set the value to None or leave it out altogether."
                raise TypeError(msg)

        #If the key does not exist, set spectral density to the default one J
        except KeyError:
            J_func = J
        #-------------------------------------------------

        #----------Computing the jump operators-----------
        t0 = ttime.time()
        if(do_local):
            jumps = jump_ops_general(local_Hamiltonian, cpl_ops, tol=tol)
        else:
            jumps = jump_ops_general(Hamiltonian_sys, cpl_ops, tol=tol)
        t1 = ttime.time()
        print("...Jumps computed in {:.2f}s. Number of jump ops: {}"
              .format(t1-t0, sum(len(jumps[i]) for i in range(len(jumps)))))
        #-------------------------------------------------

        #-----------Computing the Lamb-shift--------------
        t2 = ttime.time()
        if(include_LS):
            H_LS = Lamb_shift_general(jumps, params, J_func, psa_cut=psa_cut, tol=tol, omega_c=omega_c)
            H_LS_tot += H_LS
        t3 = ttime.time()
        print("...LS computed in {:.2f}s".format(t3-t2))
        #-------------------------------------------------

        #-----------Computing the dissipator--------------
        t4 = ttime.time()
        if(unified_eq):
            #Using unified equation approach
            if(not local_Hamiltonian==None):
                e = local_Hamiltonian.eigenenergies()
            else:
                e = Hamiltonian_sys.eigenenergies()
                
            Bohr_fs1 = np.array([b-a for a,b in list(itertools.combinations(e, 2))])
            Bohr_fs2 = np.append(Bohr_fs1, -Bohr_fs1)
            Bohr_fs = np.append(Bohr_fs2, 0)
            clusters_and_avgs = freq_clustering(Bohr_fs, width=cluster_width, plot=plot_clusters)
            diss = dissipator_unified_eq(jumps, clusters_and_avgs, params, J_func, print_info=print_info)
        else:
            #Computing the dissipator regularly
            diss = dissipator_general(jumps, params, J_func, psa_cut=psa_cut, tol=tol, omega_c=omega_c, print_info=print_info)
        t5 = ttime.time()
        print("...Dissipator computed in {:.2f}s".format(t5-t4))
        #-------------------------------------------------
        
        diss_list.append(diss)
        diss_tot += diss


    dim = Hamiltonian_sys.shape[0]
    Id = Q.qeye(dim)
    Id = Q.Qobj(Id, dims=Hamiltonian_sys.dims)
    
    #Build the unitary part of the Liouvillian
    unitary_part = -1j*(Q.tensor(Hamiltonian_sys + H_LS_tot, Id)
                        - Q.tensor(Id, (Hamiltonian_sys + H_LS_tot).trans()))

    #Setting the dimensions correct as done in the QuTiP source code
    #when constructing the liouvillian
    diss_tot = Q.Qobj(diss_tot, dims=[[Hamiltonian_sys.dims[0],
                                       Hamiltonian_sys.dims[0]],
                                      [Hamiltonian_sys.dims[1],
                                       Hamiltonian_sys.dims[1]]])
    unitary_part = Q.Qobj(unitary_part, dims=[[Hamiltonian_sys.dims[0],
                                               Hamiltonian_sys.dims[0]],
                                              [Hamiltonian_sys.dims[1],
                                               Hamiltonian_sys.dims[1]]])
    diss_list = [Q.Qobj(d, dims=[[Hamiltonian_sys.dims[0],
                                       Hamiltonian_sys.dims[0]],
                                      [Hamiltonian_sys.dims[1],
                                       Hamiltonian_sys.dims[1]]]) for d in diss_list]

    Liouvillian =  unitary_part + diss_tot

    return Liouvillian, unitary_part, diss_list

def trafo_U(Liouvillian, Nsup):
    """
    Constructs the transformation matrix U that can be utilized to
    transform the Liouvillian superoperator into a block-diagonal
    form by applying U.dag() * L * U.

    Parameters
    ----------
    Liouvillian : Qobj
        The Liouvillian superoperator of the open quantum system
    Nsup : Qobj
        A superoperator that induces a symmetry on the Liouvillian
    
    Returns
    -------
     U : Qobj
        The transformation matrix

    Notes
    -----
    The superoperator Nsup should commute with the Liouvillian. This
    ensures that it induces a symmetry, which can be utilized for the
    block diagonalization. The commutation is checked by the code.

    If [L, Nsup] == 0, the Liouvillian is block diagonalized such that
    each block is labeled by a distinct eigenvalue of Nsup. In the case
    when Nsup is corresponds to the total-number-of-quanta superoperator,
    each block is labeled by an integer telling the difference in the
    number of quanta in the ket and bra in the basis states of the Liouville
    space representation. The diagonalization itself is done by forming a
    superoperator matrix from the eigenvectors of Nsup by appending them
    next to each other in the resulting matrix.
    
    """

    #Checking that the symmetry exists. The if clause checks that all
    #elements of the commutation Qobj are zero
    commutation = Nsup*Liouvillian - Liouvillian*Nsup
    if(not np.any(commutation.full())):
        print("\nThe given superoperator commutes with the Liouvillian!")
    else:
        print("\nThe given symmetry is not correct :(")
        return

    #Computing the eigenvalues and eigenstates of the Nsup for building the
    #block structure of the Liouvillian
    N_sup_evals, N_sup_evecs = Nsup.eigenstates()
    block_labels, block_dims = np.unique(N_sup_evals.round(decimals=7), return_counts=True)

    #Building the transformation matrix from the eigenvectors of the superoperator
    U = 0*Liouvillian
    for i, vec in enumerate(N_sup_evecs):
        U.data[:,i] = vec.data

    return U

def evolution(eops,
              rho0,
              timelist,
              Liouvillian,
              check_pos_at_times=np.array([]),
              progress_prints=False):
    """
    Computes the time evolution of the initial state operator expectation values.

    Parameters
    ----------
    eops : list
        List of operators for which to compute the expectation values.
    rho0 : Qobj
        The initial state of the system as a density matrix.
    timelist : np.array
        A list of times for whicg to evaluate the dynamics.
    truncation : integer
        Truncation of the resonator Hilbert space
    check_pos_at_times : list (default [])
        A list of times for which to check the positivity of the solution.
        The positivity is measured by computing the eigenvalues of the
        density matrix and summing together the negative eigenvalues.
    progress_prints : bool (default False)
        Whether to print the progress alongside the progress bar. Printing
        is useful if the code is run in a hpc cluster, for example, to get updates
        to the log files.
    
    Returns
    -------
    results : list
        A list containing the expectation values of the wanted operators,
        the trace of the density matrix at all times and the final state.

    """
    t0 = ttime.time()
    
    rho0 = op_to_vec(rho0)

    result_dict = {"Expectation values":[],
                   "Trace":[],
                   "rho_final":Q.Qobj(),
                   "Negativities":[]}
    results = [np.empty(0) for _ in range(len(eops))]
    traces = []

    print("\nComputing evolution:")
    with alive_bar(len(timelist), force_tty=True) as bar:
        for idx, time in enumerate(timelist):

            if(progress_prints):
                if(idx%100 == 0):
                    t1 = ttime.time()
                    print("...Calculated {:.2f}%. Elapsed time {}"
                          .format(time/timelist[-1]*100,
                                  ttime.strftime("%H:%M:%S",
                                                 ttime.gmtime(t1-t0))))

            rho_t = (time*Liouvillian).expm()*rho0
            rho_t = vec_to_op(rho_t)
            trace = rho_t.tr()
            traces.append(trace)

            for i, op in enumerate(eops):
                op_t = (rho_t*op).tr()
                results[i] = np.append(results[i], op_t)

            bar()


    #Computing the negativity
    neg_list = []
    if(check_pos_at_times.any()):
        print("\nComputing the negativity of the solution at specified times")
        for time in check_pos_at_times:
            rho_t = (time*Liouvillian).expm()*rho0
            rho_t = vec_to_op(rho_t)
            neg = compute_negativity(rho_t)
            neg_list.append(neg)
            
    rho_t_final = (timelist[-1]*Liouvillian).expm()*rho0
    rho_t_final = vec_to_op(rho_t_final)

    result_dict["Negativities"] = neg_list
    result_dict["Expectation values"] = results
    result_dict["rho_final"] = rho_t_final
    result_dict["Trace"] = traces
    
    return result_dict

def compute_negativity(rho):
    
    evals = rho.eigenenergies()
    negativity = sum(e for e in evals if e < 0)
    
    return negativity

def trunc(temperature, omega_norm):
    """
    Computes the required truncation of the resonator Hilbert space.

    Parameters
    ----------
    temperature : float
        Temperature of the resonator.
    omega_norm : float
        Normalized frequency of the resonator.

    Returns
    -------
    i : integer
        The required truncation for the resonator Hilbert space.

    Notes
    -----
    Inside the function we set a tolerance parameter 'tol', which describes
    the wanted maximum population in the thermal state at the highest truncation
    of the Hilbert space. When the thermal state population is lower than the 
    tolerance, the truncation is high enough.

    """
    tol = 1e-2
    for i in range(2, 50):
        last_el = Q.thermal_dm(i, BE_distribution(omega_norm, temperature))[i-1][0][i-1]
        if(last_el < tol):
            return i
    raise ValueError("The needed truncation is too high! (over 50)")

def Gibbs(Hamilton, temperature):
    """
    Computes the Gibbs state with the given Hamiltonian.

    This is used to compare the statistical stationary state to the 
    state reached by the time evolution governed by the master equation.

    Parameters
    ----------
    Hamilton : Qobj
        The Hamiltonian for which we want to copmute the Gibbs state.
    temperature : float
        Temperature of the bath.
    
    Returns
    -------
     : Qobj
        The Qibbs state.

    """
    # beta = 1/(Boltzmann*temperature)
    # arg = -beta*hbar*omega_scale*Hamilton
    beta = 1/(temperature) #kB = 1
    arg = -beta*Hamilton #hbar = 1
    Z = arg.expm().tr()
    return 1/Z * arg.expm()
