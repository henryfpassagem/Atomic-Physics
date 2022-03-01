from arc import Calcium40 as arc_calcium
from arc.alkali_atom_functions import NumerovBack
from scipy.constants import m_e as C_m_e
from math import gamma, sqrt
import numpy as np
from atomphys import Atom, Transition

"""Monkey patch the Calcium40 class from arc to have a radial wave function. """

class Calcium40(arc_calcium):
    def __init__(self, preferQuantumDefects=True, cpp_numerov=False):
        super().__init__(preferQuantumDefects=preferQuantumDefects, cpp_numerov=cpp_numerov)

        if cpp_numerov:
            raise NotImplementedError('This does segfault for no obvious reason. Use python numerov instead.')

    def corePotential(self, l, r):
        """ core potential felt by valence electron
            For more details about derivation of model potential see
            Ref. [#marinescu]_.
            Args:
                l (int): orbital angular momentum
                r (float): distance from the nucleus (in a.u.)
            Returns:
                float: core potential felt by valence electron (in a.u. ???)
            References:
                .. [#marinescu] M. Marinescu, H. R. Sadeghpour, and A. Dalgarno
                    PRA **49**, 982 (1994),
                    https://doi.org/10.1103/PhysRevA.49.982
        """

        return -self.effectiveCharge(l, r) / r - self.alphaC / (2 * r**4) * \
            (1 - np.exp(-(r / self.rc[l])**6))

    def potential(self, l, s, j, r):
        """ returns total potential that electron feels
            Total potential = core potential + Spin-Orbit interaction
            Args:
                l (int): orbital angular momentum
                s (float): spin angular momentum
                j (float): total angular momentum
                r (float): distance from the nucleus (in a.u.)
            Returns:
                float: potential (in a.u.)
        """
        if l < 4:
            return self.corePotential(l, r) + self.alpha**2 / (2.0 * r**3) * \
                (j * (j + 1.0) - l * (l + 1.0) - s * (s + 1)) / 2.0
        else:
            # act as if it is a Hydrogen atom
            return -1. / r + self.alpha**2 / (2.0 * r**3) * \
                (j * (j + 1.0) - l * (l + 1.0) - s * (s + 1)) / 2.0

    def radialWavefunction(self, l, s, j, stateEnergy, innerLimit, outerLimit, step):
        """
        Radial part of electron wavefunction
        Calculates radial function with Numerov (from outside towards the
        core). Note that wavefunction might not be calculated all the way to
        the requested `innerLimit` if the divergence occurs before. In that
        case third returned argument gives nonzero value, corresponding to the
        first index in the array for which wavefunction was calculated. For
        quick example see `Rydberg wavefunction calculation snippet`_.
        .. _`Rydberg wavefunction calculation snippet`:
            ./Rydberg_atoms_a_primer.html#Rydberg-atom-wavefunctions
        Args:
            l (int): orbital angular momentum
            s (float): spin angular momentum
            j (float): total angular momentum
            stateEnergy (float): state energy, relative to ionization
                threshold, should be given in atomic units (Hatree)
            innerLimit (float): inner limit at which wavefunction is requested
            outerLimit (float): outer limit at which wavefunction is requested
            step (flaot): radial step for integration mesh (a.u.)
        Returns:
            List[float], List[flaot], int:
                :math:`r`
                :math:`R(r)\\cdot r`
        .. note::
            Radial wavefunction is not scaled to unity! This normalization
            condition means that we are using spherical harmonics which are
            normalized such that
            :math:`\\int \\mathrm{d}\\theta~\\mathrm{d}\\psi~Y(l,m_l)^* \
            \\times Y(l',m_{l'})  =  \\delta (l,l') ~\\delta (m_l, m_{l'})`.
        Note:
            Alternative calculation methods can be added here (potenatial
            package expansion).
        """
        innerLimit = max(
            4. * step, innerLimit)  # prevent divergence due to hitting 0
        if self.cpp_numerov:
            # Does not work!
            # efficiant implementation in C
            if (l < 4):
                d = self.NumerovWavefunction(
                    innerLimit, outerLimit,
                    step, 0.01, 0.01,
                    l, s, j, stateEnergy, self.alphaC, self.alpha,
                    self.Z,
                    self.a1[l], self.a2[l], self.a3[l], self.a4[l],
                    self.rc[l],
                    (self.mass - C_m_e) / self.mass)
            else:
                d = self.NumerovWavefunction(
                    innerLimit, outerLimit,
                    step, 0.01, 0.01,
                    l, s, j, stateEnergy, self.alphaC, self.alpha,
                    self.Z, 0., 0., 0., 0., 0.,
                    (self.mass - C_m_e) / self.mass)
            print(d)
            psi_r = d[0]
            r = d[1]
            suma = np.trapz(psi_r**2, x=r)
            psi_r = psi_r / (sqrt(suma))
        else:
            # full implementation in Python
            mu = (self.mass - C_m_e) / self.mass

            def potential(x):
                r = x * x
                return -3. / (4. * r) + 4. * r * (
                    2. * mu * (stateEnergy - self.potential(l, s, j, r))
                    - l * (l + 1) / (r**2)
                    )

            r, psi_r = NumerovBack(innerLimit, outerLimit, potential,
                                   step, 0.01, 0.01)

            suma = np.trapz(psi_r**2, x=r)
            psi_r = psi_r / (sqrt(suma))

        return r, psi_r


class AtomphysCalciumPatch(Atom):
    """Patch of Atom('Ca') including the transitions 
        (4s4p {^1\mathrm{P}_1}\) -  (4s4d {^1\mathrm{D}_2}\) at 732.8 nm
        (4s4p {^1\mathrm{P}_1}\) -  (4s5s {^1\mathrm{S}_0}\) at 1034.4 nm
    """

    def __init__(self, ureg=None, refresh_cache=False):
        super().__init__(atom='Ca', ureg=ureg, refresh_cache=refresh_cache)

        # Add the missing transitions
        # Define the states 4s4p ^1P_1, 4s4d ^1D_2 and 4s5s ^1S_0
        for state in self.states:
            if state.configuration == '3p6.4s.4p' and state.term == '1P1':
                Ca_4s4p = state
                continue
            if state.configuration == '3p6.4s.4d' and state.term == '1D2':
                Ca_4s4d = state
                continue
            if state.configuration == '3p6.4s.5s' and state.term == '1S0':
                Ca_4s5s = state
                continue

        # Linewidth of 4s4d ^1D_0 from https://www.ptb.de/cms/fileadmin/internet/fachabteilungen/abteilung_4/4.3_quantenoptik_und_laengeneinheit/4.31/Dissertation_Binnewies_2001.pdf
        Gamma_4s4d = 1.4e7 * self.units('Hz')
        # Linewidth of 4s5s ^1S_0 from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.67.043407
        Gamma_4s5s = 2*np.pi * 4.77 * self.units('MHz')

        # Add the transition to the transition registry. 
        # This assumes that the decay from both states is primarily to 4s4p ^1P_1
        self.transitions.append(Transition(state_i=Ca_4s4p, state_f=Ca_4s4d, Gamma=Gamma_4s4d))
        self.transitions.append(Transition(state_i=Ca_4s4p, state_f=Ca_4s5s, Gamma=Gamma_4s5s))


