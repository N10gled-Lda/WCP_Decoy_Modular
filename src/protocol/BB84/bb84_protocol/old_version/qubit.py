### qubit.py ###
# This file implements the qubit generation and their functions for the BB84 protocol.

import numpy as np
import random

class Gates:
	"""Class to represent commun bases and gates."""
	X = np.matrix([[0, 1], [1, 0]])
	Y = np.matrix([[0, -1j], [1j, 0]])
	Z = np.matrix([[1, 0], [0, -1]])
	H = (1/np.sqrt(2))*np.matrix([[1,1],[1,-1]])
	S = np.matrix([[1, 0], [0, 1j]])
	I = np.matrix([[1, 0], [0, 1]])

class Qubit():
	"""Class to represent a qubit."""
	def __init__(self,initial_state):
		'''Initializes the qubit in the given state/bit.'''
		if initial_state: 	# |1>
			self.__state = np.matrix([[0],[1]])
		else: 				# |0>
			self.__state = np.matrix([[1],[0]])
		self.__measured = False

	def show(self):
		aux = ""
		if round((np.matrix([1,0])*self.__state).item(),2):
			aux += "{0}|0>".format(str(round((np.matrix([1,0])*self.__state).item(),2)) if round((np.matrix([1,0])*self.__state).item(),2) != 1.0 else '')
		if round((np.matrix([0,1])*self.__state).item(),2):
			if aux:
				aux += " + "
			aux += "{0}|1>".format(str(round((np.matrix([0,1])*self.__state).item(),2)) if round((np.matrix([0,1])*self.__state).item(),2) != 1.0 else '')
		return aux
	
	def measure(self):
		if self.__measured:
			raise Exception("Qubit already measured!")
		M = 1000000
		m = random.randint(0,M-1)
		self.__measured = True
		if m < round(pow(((np.matrix([1,0])*self.__state).item()),2),2)*M:
			return 0
		else:
			return 1

    # def measure(self, observable):
    #     # check that the observable is Hermitian
    #     if not np.allclose(observable, observable.conj().T):
    #         raise ValueError('Observables must be Hermitian matrices')

    #     # perform a projective measurement with the given Hermitian
    #     e_val, e_vect = np.linalg.eig(observable)
    #     p = list(map(lambda v: abs(v) ** 2, np.matmul(e_vect.T, self.state)))
    #     if not np.isclose(sum(p), 1):
    #         raise ValueError('Probabilities do not sum to 1')
    #     rand = secrets.SystemRandom().random()
    #     pcum = enumerate(np.cumsum(p))
    #     i = next((n[0] for n in pcum if n[1] > rand), len(p) - 1)

    #     # place into state e_vect[i], normalized by the inner product p[i]
    #     outstate = e_vect.T[i] / np.linalg.norm(e_vect.T[i])
    #     out = Qubit(zero=outstate[0], one=outstate[1])
    #     return e_val[i], out

	def hadamard(self):
		'''Apply the Hadamard gate to the qubit.'''
		if self.__measured:
			raise Exception("Qubit already measured!")
		self.__state = Gates.H*self.__state

	def X(self):
		if self.__measured:
			raise Exception("Qubit already measured!")
		self.__state = Gates.X*self.__state

	def gate(self, gate: Gates):
		if self.__measured:
			raise Exception("Qubit already measured!")
		# Apply the gate to the qubit
		self.__state = gate*self.__state
