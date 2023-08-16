import numpy as np
from matplotlib import pyplot as plt
import functools as ftls
from scipy import linalg
import copy
import statistics as st
from matplotlib.pyplot import *

#Initialize Pauli Matrices
def pauli_matrices(n):
    if n == 1:
        s1 = np.array([[0.0, 1.0], [1.0, 0.0]])
        return s1
    if n ==2:
        s2 = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype = 'complex')
        return s2
    if n == 3:
        s3 = np.array([[1.0, 0.0], [0.0, -1.0]])
        return s3

#Initialize psi
def initial_psi(qubits):
    psi = np.ones(2**qubits, dtype='complex128')/np.sqrt(2**qubits)
    return psi

#Calculate the commutator of two matrices
def commutator(hamiltonian_d, hamiltonian_p, anti = False):
    if anti:
        return hamiltonian_d @ hamiltonian_p + hamiltonian_p @ hamiltonian_d
    else:
        return hamiltonian_d @ hamiltonian_p - hamiltonian_p @ hamiltonian_d

#Calculate eigenvalues of a matrix
def eigenvalue(matrix):
    eigenval = np.linalg.eig(matrix)[0]
    return eigenval

#Calculate eigenvectors of a matrix
def eigenvector(matrix):
    eigenvec = np.linalg.eig(matrix)[1]
    return eigenvec

#Calculate the expectation value of a commutator with a given state
def expectation_value(commutator, psi):
    normalized_psi = psi/np.linalg.norm(psi)
    conjugated = np.conj(normalized_psi)
    transversed = conjugated.T
    rho = np.outer(normalized_psi, conjugated)
    expectation2 = np.trace(rho*commutator)
    matrix_mul = (commutator) @ normalized_psi
    expectation = np.inner(transversed, matrix_mul)
    return expectation, expectation2

#Generate Problem Hamiltonian
def Hamiltonian_problem(qubits, operator):
    number_of_qubits = len(qubits)
    Hamilt = np.zeros([(2**number_of_qubits),(2**number_of_qubits)], dtype='complex128') #Array 2^#ofQubits x 2^#ofQubits
    identity = np.eye((2**number_of_qubits), (2**number_of_qubits), dtype='complex128') #Diagonal array 2^#ofQubits x 2^#ofQubits
    for j, k in zip(edges):
        j = np.sort(j)
        Hamilt += (-1.0) * (identity - (1.0*k*ftls.reduce(np.kron, [np.eye(2**j[0]), (operator), np.eye(2**(j[1]-j[0]-1)), (operator), np.eye(2**(number_of_qubits-j[1]-1))])))/2
    if np.isreal(Hamilt).all():
        Hamilt = Hamilt.real
    return Hamilt

#Generate Driver Hamiltonian
def Hamiltonian_driver(qubits, operator):
    number_of_qubits = len(qubits)
    Hamilt = np.zeros([(2**number_of_qubits),(2**number_of_qubits)], dtype='complex128')
    for j in (qubits):
        Hamilt += 1.0 * ftls.reduce(np.kron, [np.eye(2**j), (operator), np.eye(2**(number_of_qubits-j-1))])  
    if np.isreal(Hamilt).all():
        Hamilt = Hamilt.real
    return Hamilt, number_of_qubits

#Time Evolution Function
def time_evolution(hamiltonian, num_layers, psi_initial, qubits, beta = 0.0, dt = 0.01):
    qubits = len(qubits)
    unitary = np.eye(2**qubits, dtype='complex128')
    beta_list = [beta]
    h_p = hamiltonian[0]
    h_d = hamiltonian[1]
    psi_list = []
    cmtr = commutator(h_d, h_p)
    h_p_list = []
    unitary_list = []
    psi = copy.deepcopy(psi_initial)
    psi_list.append(psi)
    for n in range(num_layers):
        psi = linalg.expm(((-1.0j) * 1.0 * h_p *  dt)) @ psi
        psi = linalg.expm(((-1.0j) * beta_list[-1] * h_d  * dt)) @ psi
        psi_list.append(psi)
        h_p_ev = expectation_value(h_p, psi)[0].real
        h_p_list.append(h_p_ev)
        beta_updt = (-1.0j * expectation_value(cmtr, psi)[0]).real
        beta_list.append(beta_updt)
    return beta_list, psi, unitary, h_p_list, unitary_list, psi_list

#Generate the probability of obtaining the global minimum solution to the original combinatorial optimization problem
def phi(psi, H_p_argmin):
    phi = 0.0
    phi_mean = 0.0
    phi_list = []
    initial_qubit = np.eye(psi[0].shape[0], dtype = 'complex128')
    for x in range(len(psi)):
        psi_x = psi[x]
        phi = 0.0
        qubit = (initial_qubit)
        for i in H_p_argmin:
            phi = phi + (np.absolute(np.vdot(psi_x, qubit[:,i])))**2
            phi_mean = phi_mean + (np.absolute(np.vdot(psi_x, qubit[:,i])))**2
        phi_list.append(phi)
    phi_mean = phi_mean/len(psi)
    return phi_mean, phi_list

#Plot the approximation ratio and the probability of sampling the global optimum as a function of the number of 
#layers for a single Hamiltonian.
def plot_falqon(problem_hamiltonian_expectation_value, h_p, psi_n, H_p_argmin, graph):
    h_p_min = min(np.linalg.eig(h_p)[0])
    print(h_p_min)
    r = []
    for i in range(len(problem_hamiltonian_expectation_value)):
        value = (problem_hamiltonian_expectation_value[i])/h_p_min
        r.append(value)
    print('Approximation mean ratio:\n',st.mean(r))
    print('Minimum expectation value of problem Hamiltonian: ', h_p_min)
    plt.plot(r, label='approximation ratio')
    plt.axhline(y=0.93, color='black', linestyle='--')
    plt.grid(visible=True, which='major', axis='both')
    plt.xlabel("Layers")
    phi_number = phi(psi_n, H_p_argmin)
    print('Mean phi value = ', phi_number[0])
    plt.axhline(y=0.25, color='black', linestyle='-')
    plt.plot(phi_number[1], label='Probability of sampling global optimum')
    plt.title('Performance of FALQON for %s -regular graph with %s vertices.'%(graph[1], graph[2]))
    plt.legend(fontsize='medium', loc = 4)
    plt.show()

#Plot the beta values at each layer
def plot_Beta(beta_list_original):
    abs_beta = []
    for i in range(len(beta_list_original)):
        abs_beta.append(-1.0*(beta_list_original[i]))
    plt.plot(abs_beta,label='beta')
    plt.legend(fontsize='medium', loc = 1) #Shows the lenged 
    plt.grid(visible=True, which='major', axis='both')
    plt.xlabel("Layers")
    plt.ylabel(r'$\beta$')
    plt.show()

def execute_FALQON(layers, qubits, beta, dt):
    h_p_values = [] #Holds the Problem Hamiltonian values
    psi_values = [] #Holds the psi values
    H_p_argmin_values = [] #Holds the Problem Hamiltonian argument values
    Hamiltonian_problem_list = [] #List of Problem Hamiltonians
    Hamiltonian_driver_list = [] #List of Driver Hamiltonians
    graphs_array = [] #Array of Graphs
    beta_values = [] #Holds the Beta values
    graph_for_FALQON =[] #List of Graphs for FALQON to run on
    for g in range(layers):
        z = pauli_matrices(3) #initializes z pauli matrix
        H_p = Hamiltonian_problem(graph_for_FALQON[0].edges, graph_for_FALQON[0].nodes, z, weighted = False) #Creates problem Hamiltonian
        x = pauli_matrices(1) #Initializes x pauli matrix
        H_d = Hamiltonian_driver(graph_for_FALQON[0].nodes, x) #Creates Driver Hamiltonian
        psi_initial = initial_psi(len(qubits)) #Initializes psi with the number of qubits
        hamiltonian_preparation = []*2 #Creates Hamiltonian preparation array that holds the problem Hamiltonian
        hamiltonian_preparation.append(H_p[0]) #Add the problem Hamiltonian
        hamiltonian_preparation.append(H_d[0]) #Add the driver Hamiltonian
        test_time_evolution = time_evolution(hamiltonian_preparation, layers, psi_initial, graph_for_FALQON[0].nodes, beta, dt) #Do the time evolution
        H_p_argmin = np.where((np.diag(H_p[0])-np.diag(H_p[0])[np.argmin(np.diag(H_p[0]))])<=1e-10)[0]
        h_p_values.append(test_time_evolution[3]) #adds the 4th element from time_evolution to problem hamiltonian values
        psi_values.append(test_time_evolution[5]) #adds the 6th element from time_evoltuion to psi values
        H_p_argmin_values.append(H_p_argmin) #adds the problem hamiltonian argument value
        Hamiltonian_problem_list.append(H_p[0]) #adds the 1st element of the problem hamiltonian to the list
        Hamiltonian_driver_list.append(H_d[0]) #adds the 1st element of the driver hamiltonian to the list
        graphs_array.append(graph_for_FALQON) #adds the FALQON graph to the graph array
        plot_falqon(test_time_evolution[3], H_p[0], test_time_evolution[5], H_p_argmin, graph_for_FALQON) #plots FALQON
        beta_list_original = test_time_evolution[0] #creats beta list and adds 1st element of time evolution to it
        beta_values.append(beta_list_original) #adds the beta list to beta values
        plot_Beta(beta_values[g]) #plot beta values
    return h_p_values, Hamiltonian_problem_list, psi_values, H_p_argmin_values, graphs_array, beta_values, Hamiltonian_driver_list

execute_FALQON(250, 2, 0, 0.0425)