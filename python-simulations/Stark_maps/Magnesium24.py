from arc import DivalentAtom, physical_constants

# IMPORTANT: Make sure to run patch_arc.sh the first time you use this class.

class Magnesium24(DivalentAtom):
    """
    Properties of Magnesium 24 atoms
    """

    # Core polarizbility
    # Used as parameter in AlkaliAtom.corePotential 
    # and as inner integration limit in AlkaliAtom.radialWavefunction (r_i = (\alpha_C})^{-3})
    alphaC = 0.49 # (a.u.) Ref. E. Luc-Koenig et al., J. Phys. B, 30 (1997) 5213-5232
    
    # Model potential paremeters. Used in AkaliAtom.effectiveCharge. Only used if l < 4.
    # Ref. E. Luc-Koenig et al., J. Phys. B, 30 (1997) 5213-5232
    a1 = [4.51367, 4.71475, 2.99158, 2.99158]
    a2 = [11.81954, 10.71581, 7.69976, 7.69976]
    a3 = [2.97141, 2.59888, 4.38828, 4.38828]
    a4 = [0.0, 0.0, 0.0, 0.0] # r^2 term
    rc = [1.44776, 1.71333, 1.73093, 1.73093]
    
    ionisationEnergy = 7.646236       #: (eV) source NIST Atomic Spectra Database Ionization Energies

    Z = 12
    I = 0.
    
    #: source NIST, Atomic Weights and Isotopic Compositions [#c14]_
    mass =  23.985041697 * physical_constants["atomic mass constant"][0]
    reduced_mass = mass / (mass + physical_constants['electron mass'][0])
    scaledRydbergConstant = physical_constants['Rydberg constant'][0] * reduced_mass * physical_constants["inverse meter-electron volt relationship"][0]

    
    #: source S.F. Dyubko et al., CAOL*2013 International Conference on Advanced Optoelectronics & Lasers (2013)
    # quantum defects format : [delta_0, delta_2, ..., delta_10]
    quantumDefect = [[[1.525367, -0.0310, 1.364, -3.37, 0, 0], # quantum defects for 1S0
                      [1.051333, -0.3679, 0.874, -3.51, 0, 0], # quantum defects for 1P1
                      [0.612110, -3.147, 8.25, -5.54, 0, 0],   # quantum defects for 1D2
                      [0.052167, -0.253, 2.64, -29, 0, 0],     # quantum defects for 1F3
                      [0.014971, -0.540, 87, 0, 0, 0]],        # quantum defects for 1G4
                     [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                     [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                     [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    
    """ Contains list of modified Rydberg-Ritz coefficients for calculating
        quantum defects for
        [[ :math:`^1S_{0},^1P_{1},^1D_{2},^1F_{3},^1G_{4},`]]."""
    
    groundStateN = 3
    minQuantumDefectN = 7
    preferQuantumDefects = False
    
    # levels that are for smaller n than ground level, but are above in energy
    # due to angular part
    extraLevels = []

    levelDataFromNIST = "mg_level_data.csv"

    precalculatedDB = "mg_precalculated.db"
    dipoleMatrixElementFile = "mg_dipole_matrix_elements.npy"
    quadrupoleMatrixElementFile = "mg_quadrupole_matrix_elements.npy"
    
    # TODO
    literatureDMEfilename = ''#'magnesium_literature_dme.csv'

    elementName = 'Mg24'

    meltingPoint = 650 + 273.15  #: in K

    #: Quantum defect principal quantum number fitting ranges for different
    #: series TODO
    defectFittingRange = {"1S0": [5, 80], "3S1": [13, 45], "1P1": [4, 80],
                          "3P2": [8, 18], "3P1": [8, 22], "3P0": [8, 15],
                          "1D2": [3, 80], "3D3": [20, 45], "3D2": [22, 37],
                          "3D1": [20, 32], "1F3": [4, 80], "3F4": [10, 24],
                          "3F3": [10, 24], "3F2": [10, 24]}
    
    def getPressure(self, temperature):
        raise NotImplementedError()

