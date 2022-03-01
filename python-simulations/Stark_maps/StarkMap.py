import numpy as np
from scipy.optimize import fsolve
from arc import *
from .ureg import ureg


def get_ns(W: float) -> float:
    """Calculate the effective quantum number n*. """
    
    if W.is_compatible_with('GHz'):
        W = W * ureg('h')
    elif W.is_compatible_with('1/cm'):
        W = W * ureg('h') * ureg('c')
    
    #print(W, W.is_compatible_with(1 * ureg('GHz')))
    W_au = W.to('atomic_unit_of_energy')
    ns = 1 / np.sqrt(-2 * W_au.magnitude)
    
    return ns


def get_classical_limit(W: float, m: int):
    """Calculate the ionization field for a state of energy W. 
    
    We use the method described in Cooke and Gallagher PRA 17, 3 (1978)
    """

    # Get the effective quantum number
    ns = get_ns(W)
    
    if m == 0:
        # Use the solution x = 1 for m = 0
        Fth = 1 / (16 * ns**4) * ureg('atomic_unit_of_electric_field')
    else:
        # Solve the equation for m =/= 0
        f = lambda x: (1 + x) * np.sqrt(1 + 3*x**4) * (1 - x) * (1 + x**2) - np.abs(m) / ns
        x = fsolve(f, x0=1)
        x = float(x)

        Fth = x**2 / ((1 + 3*x**4)**2 * ns**4) * ureg('atomic_unit_of_electric_field')
    
    return Fth


def get_path_to_ionization(m: int, E: np.array, W: np.array):
    """Get the field and energy up to the ionization point. """

    # Get the number of electric field values
    N = len(E)
    
    for i in range(N):
        # Calculate the ionization field
        limiting_field = get_classical_limit(W[i], m)
        # Stop when we are above
        if limiting_field < E[i]:
            break
            
    return E[:i], W[:i]


class ModifiedStarkMap:
    def __init__(self, atom, nmin, nmax, lmax, m, s=0):
        """Constructor. """
        
        # We don't care about n, l and j
        n = nmin; l = 0; j = l
        # We only care about m because it defines which states define the basis
        m = np.abs(m)

        # Initialize the basis states
        self.calc = StarkMap(atom)
        self.calc.defineBasis(n, l, j, m, nmin, nmax, lmax, progressOutput=False, s=s)
        
    def diagonalize(self, E: float) -> tuple:
        """Diagonalize the Stark Hamiltonian for a field E. """
                
        # Construct the Stark Hamoltonian
        matrix = self.calc.mat1 + self.calc.mat2 * E.to('V/m').magnitude
        
        # Diagonalize. eigh sorts the eigenvalues and corresponding eigenvectors in ascending order.
        ev, evec = np.linalg.eigh(matrix)
        
        # Put unit on energy
        ev = ev * ureg('GHz')
            
        return ev, evec
    
    def calculate_Stark_map(self, E: np.array) -> np.array:
        """Calculate the Stark map for all values of E. 
        
        # Arguments
        * E::np.array(N) - Electric field values.
        
        # Returns
        * W::np.array(M, N) - Energy of all M basis states.
        """
        
        # Allocate memory for the result
        W = []
        
        for Ei in E:
            # Calculate the energy for each field value
            energy, _ = self.diagonalize(Ei)
            # Save the energy
            W.append(energy.to('GHz').magnitude)
            
        # Turn result into matrix and put unit back on
        W = np.array(W) * ureg('GHz')
        
        # Transpose the result s.t. each row corresponds to the energy W(E) of one particular state
        W = W.T
        
        return W
    
    def get_basis_state_index(self, state_label: tuple) -> int:
        """Get the index of a basis element. 
        
        # Arguments
        * state_label - (n, l, m) or (n, l, j, mj).
        
        # Returns
        * state_index::int - Index of the state in self.calc.basisStates
        """
        
        # Extract the state
        if len(state_label) == 3:
            n, l, m = state_label
            j = l
        elif len(state_label) == 4:
            n, l, j, m = state_label
        else:
            raise ValueError('A state label must have 3 or 4 elements.')
        
        # Find it within the basis states
        try:
            return self.calc.basisStates.index([n, l, j, m])
        except ValueError:
            raise ValueError(f'State ({n}, {l}, {j}, {m}) is not a basis state.')
            
    def get_basis_state_vector(self, state_label: tuple) -> np.array:
        """Get the vector representing a basis state. 
        
        # Arguments
        * state_label - (n, l, m) or (n, l, j, mj).
        
        # Returns
        * state_vector::np.array(M) - Vector representation of the state.
        """
        
        # Get the index of the basis state
        state_index = self.get_basis_state_index(state_label)
        
        # Get total number of basis states
        M = len(self.calc.basisStates)
        
        # Define the state vector
        state_vector = np.zeros(M)
        state_vector[state_index] = 1
        
        return state_vector
    
    def get_overlap(self, state_label: tuple, states: np.array) -> np.array:
        
        # Get the basis vector of the state
        state_vector = self.get_basis_state_vector(state_label)
        # Calculate the overlap with all states
        overlap = np.dot(state_vector.T, states)**2
        
        return overlap
    
    def find_non_hydrogenic_states(self, n: int, m: int, s: float = 0, debug: bool = False):
        """Identify the non-hydrogenic states of principal quantum number n.
        
        The non-hydrogenic states of a manifold are the ones that are non-degenerate at zero field.
        They can be identified by calculating the overlap of the eigenstates of the Hamiltonian with 
        the uncoupled basis states at zero field. If the overlap is unity they are non-degenerate and we
        can assign a (n, l, m) or (n, l, j, m) label to that state.
        
        # Arguments
        * n::int - Principal quantum number of the state manifold
        * m::int - Magnetic quantum number of the states in the manifold.
        * debug::bool - Additional output.
        
        # Returns
        (state_labels::list, state_indices::list) - Tuple of lists. 
                                                    * `state_labels` contains the (n, l, m) labels of all 
                                                        non-degenerate states
                                                    * `state_indices` contains the indices of the eigenvalues
                                                        that correspond to these states.
        """
        
        # Get minimum and maximum value of l
        l_min = np.abs(m); l_max = n - 1
        # Get all allowed values of l
        ls = np.arange(l_min, l_max + 1, 1)
        
        if s == 0:
            # Get absall possible (n, l, m) labels
            possible_states = [(n, l, m) for l in ls]
        elif s == 0.5:
            possible_states = []
            for l in ls:
                jmin = np.abs(l - s)
                jmax = l + s
                for j in np.arange(jmin, jmax + 1, 1):
                    if np.abs(m) <= j:
                        possible_states.append((n, l, j, m))
        else:
            raise NotImplementedError(f'Spin <{s}> not implemented.')
        
        # Get the atom class
        atom = self.calc.atom
        
        # Diagonalize the Hamiltonian at (almost) zero field (at F = 0 the calculation is numerically unstable)
        ev, _ = self.diagonalize(0.001 * ureg('V/cm'))
        
        # Allocate memory for the result
        state_labels = []
        state_indices = []
        
        # Iterate over all possible states
        for state in possible_states:
            if len(state) == 3:
                # turn (n, l, m) into (n, l, j=l, s)
                nljs_state = (state[0], state[1], state[1], 0)
            else:
                # turn (n, l, j, m) into (n, l, j, s)
                nljs_state = (state[0], state[1], state[2], 0.5)
                
            # A state is non-hydrogenic if it has a non-vanishing quantum defect
            if atom.getQuantumDefect(*nljs_state) > 0:
                # Get the energy of the state in GHz
                energy = atom.getEnergy(*nljs_state) * ureg('eV / h')

                # Get the closest energy
                i = np.argmin(np.abs(energy - ev))
                
                # Save the state
                state_labels.append(state)
                state_indices.append(i)
                
                if debug:
                    print(state, i, (energy / ureg('c')).to('1/cm'), (ev[i] / ureg('c')).to('1/cm'))
        
        # reverse order s.t. state with highest energy is first
        state_labels = state_labels[::-1]
        state_indices = state_indices[::-1]
        
        return (state_labels, state_indices)

    def find_non_hydrogenic_state_index(self, state_label: tuple) -> int:
        n = state_label[0]
        m = state_label[-1]
        if len(state_label) == 3:
            s = 0
        else:
            s = 0.5
        
        nh_state_labels, nh_basis_indices = self.find_non_hydrogenic_states(n, m, s)
        
        index = nh_state_labels.index(state_label)
        
        return nh_basis_indices[index]
    
    def find_hydrogenic_states(self, n: int, m: int, debug: bool = False):
        """Identify the hydrogenic states of principal quantum number n.
        
        The hydrogenic states of a manifold are the ones that are degenerate at zero field.
        The highest energy state (n, n1_max, 0, m) of a manifold can be identified by finding the state closest
        to the calculated first order energy shift at low field.
        We evaluate the Stark shift at 1% of the Ignes-Teller limit (where states of different manifolds start 
        crossing) in order to avoid miss-assignment. Also note that the first order shift is always overestimating 
        the energy because the second order term has a negative sign.
        
        Once the highest state in the manifold is identified, the states below can be assigned easily since
        their energy decreases monotonically with increasing n2. As soon as we find the (n, n1, n2, m) label
        of the higest energy non-degenerate state (n, l_max, m) we identify the remaining (n, n1, n2, m) states
        by the indices we already found for the non-degenerate states. This is necessary to avoid miss-assignment
        because low l states might be below the n-1 hydrogenic manifold.
        
        # Arguments
        * n::int - Principal quantum number of the state manifold
        * m::int - Magnetic quantum number of the states in the manifold.
        * nh_state_indices::list - Indices of the eigenvalues of the non-degenerate states
        * debug::bool - Additional output.
        
        # Returns
        (state_labels::list, state_indices::list) - Tuple of lists. 
                                                    * `state_labels` contains the (n, n1, n2, m) labels of all 
                                                       states in the n manifold.
                                                    * `state_indices` contain the indices of the eigenvalues that
                                                       correspond to these states.
        """
        
        # Get maximum value of n1. This is the state with the highest energy in the manifold.
        n1_max = n - np.abs(m) - 1
        
        # Allocate memory for the result.
        state_labels = [(n, n1, n - n1 - m - 1, m) for n1 in np.arange(n1_max, -1, -1)]
        state_indices = []
        
        # Get the zero field energy of the hydrogenic energy levels
        E0 = self.calc.atom.getEnergy(n, n-1, n-1, s=0) * ureg.eV
        
        # Calculate n*
        ns = get_ns(E0)
        
        # Evaluate states at 1% of the Ignis-Teller-Limit
        F = 0.01 * (1 / (3*ns**5) * ureg('a_u_electric_field')).to('V/cm')
        ev, _ = self.diagonalize(F)
        
        # Get first order Stark shift for the outermost state (n1, n2) = (n1_max, 0)
        dE_max = (3 * F.to('a_u_electric_field').magnitude / 2 * ns * n1_max) * ureg('a_u_energy')
        E_n1_max = E0 + dE_max
        
        # Get index of state (n, n1_max, 0, m) i.e. the state closest to the first order perturbation result
        i0 = np.argmin(np.abs(ev - E_n1_max / ureg.h))
        if debug:
            E_calc = (E_n1_max / ureg('h * c')).to('1/cm')
            E_nearest = (ev[i0] / ureg('c')).to('1/cm')
            print(i0, E_calc, E_nearest)
        
        # Iterate over all (n, n1, n2, m) states
        for (i, state) in enumerate(state_labels):
            # Get the index of the eigenvalue that corresponds to that state
            index = i0 - i
            state_indices.append(index)
            
            if debug:
                print(state, index, (ev[index] / ureg('c')).to('1/cm'))
            """
            # Stop when we find the (n, n1, n2, m) label of the first non-degenerate state (n, lmax, m)
            if index in nh_state_indices:
                break
            else:
                state_indices.append(index)
            """
        
        # Add the indices of the eigenvalues of the non-degenerate states
        #state_indices += nh_state_indices
                    
        return (state_labels, state_indices)
    
    def find_state_labels(self, n: int, m: int, debug: bool = False):
        # Get the non-hydrogenic states
        nh_labels, nh_indices = self.find_non_hydrogenic_states(n, m, debug=debug)
        # Get the hydrogenic states
        h_labels, h_indices = self.find_hydrogenic_states(n, m, debug=debug)
        
        # Replace the wrongly assigned states
        n_nh_states = len(nh_labels)
        
        if n_nh_states > 0:
            state_labels = h_labels[:-n_nh_states] + nh_labels
            state_indices = h_indices[:-n_nh_states] + nh_indices
        else:
            state_labels = h_labels
            state_indices = h_indices
        
        # Make sure we didn't miss any state
        assert len(state_labels) == len(h_labels)
        
        return state_labels, state_indices