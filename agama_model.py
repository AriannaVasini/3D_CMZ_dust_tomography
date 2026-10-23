import numpy as np
import agama


# =================================================================================
# AGAMA model of the NSD (Sormani et al. 2022, MNRAS 512, 1857)
# =================================================================================

def build_agama_model():
    """
    Self-consistent DF-based model of the nuclear stellar disk (NSD) with a fixed nuclear star cluster (NSC).
    Returns the GalaxyModel of the NSD (potential, DF and action finder are gm.potential, gm.df, gm.af).
    """
    agama.setUnits(length=1, velocity=1, mass=1e10)   # 1 kpc, 1 km/s, 1e10 Msun

    model = agama.SelfConsistentModel(
        RminCyl=0.005, RmaxCyl=1.0, sizeRadialCyl=25,
        zminCyl=0.005, zmaxCyl=1.0, sizeVerticalCyl=25,
        RminSph=0.0001, RmaxSph=0.1, sizeRadialSph=25,
        lmaxAngularSph=8
    )

    density_NSC = agama.Density(
        type='Dehnen', mass=6.1e-3, gamma=0.71,
        scaleRadius=5.9e-3, axisRatioZ=0.73
    )

    d1 = agama.Density(type='Spheroid', DensityNorm=0.9*222.885,
                       gamma=0, beta=0, axisRatioZ=0.37,
                       outerCutoffRadius=0.0050617, cutoffStrength=0.7194)
    d2 = agama.Density(type='Spheroid', DensityNorm=0.9*169.975,
                       gamma=0, beta=0, axisRatioZ=0.37,
                       outerCutoffRadius=0.0246, cutoffStrength=0.7933)

    density_NSD = agama.Density(d1, d2)

    model.components.append(agama.Component(density=density_NSC, disklike=False))
    model.components.append(agama.Component(density=density_NSD, disklike=True))
    model.iterate()

    dfNSD = agama.DistributionFunction(
        potential=model.potential,
        type='QuasiIsothermal',
        mass=0.097, Rdisk=0.075, Hdisk=0.025,
        sigmar0=75.0, Rsigmar=1.0, sigmamin=2.0, Jmin=10.0
    )

    model.components[1] = agama.Component(
        df=dfNSD, disklike=True,
        RminCyl=0.005, RmaxCyl=0.75,
        sizeRadialCyl=20, zminCyl=0.005,
        zmaxCyl=0.25, sizeVerticalCyl=15
    )

    # iterate to make NSD DF & potential self-consistent
    for iteration in range(1,5):
        print('Starting iteration #%d' % iteration)
        model.iterate()

    gm = agama.GalaxyModel(model.potential, dfNSD)
    return gm


# =================================================================================
# Stars of one los
# =================================================================================

def generate_stars(gm, l_pc, b_pc, y_min, y_max, n_stars_pool, path_stars_pool, cell_size_pc=10.0, vmax=400.0):
    """
    Samples n_stars_pool stars of the NSD inside the cell centred on (l_pc, b_pc) (cell_size_pc x cell_size_pc,
    along the los between y_min and y_max), directly from the DF of the model
    (agama.sampleNdim in the 6D box: cell x velocities in [-vmax, vmax] km/s).
    The seed depends on the los, so the stars of a los are always the same.
    In the frame of AGAMA the los is the x axis and l the y axis; in the saved array (and in the paper) the los is y.
    Saves an array with columns: y (kpc, along the los), x (kpc, along l), z (kpc, along b), id, v_l (km/s).
    """
    agama.setRandomSeed(1000*int(l_pc + 1000) + int(b_pc + 1000))   # positive and different for every los (0 would take the current time)

    l_min_kpc, l_max_kpc = (l_pc - cell_size_pc/2) * 0.001, (l_pc + cell_size_pc/2) * 0.001
    b_min_kpc, b_max_kpc = (b_pc - cell_size_pc/2) * 0.001, (b_pc + cell_size_pc/2) * 0.001

    def df_xv(xv):
        # value of the DF at the points (x, v) of AGAMA; unbound orbits have no actions (nan) -> 0
        return np.nan_to_num(gm.df(gm.af(xv)), nan=0.0)

    xv, _, _, _ = agama.sampleNdim(df_xv, n_stars_pool,
                                   [y_min*0.001, l_min_kpc, b_min_kpc, -vmax, -vmax, -vmax],
                                   [y_max*0.001, l_max_kpc, b_max_kpc,  vmax,  vmax,  vmax])

    stars_pool = np.zeros([n_stars_pool, 5])
    stars_pool[:,0] = xv[:,0]   # along the los
    stars_pool[:,1] = xv[:,1]   # along l
    stars_pool[:,2] = xv[:,2]   # along b
    stars_pool[:,3] = np.arange(n_stars_pool)
    stars_pool[:,4] = xv[:,4]   # velocity along l (v_l)

    np.save(path_stars_pool, stars_pool)
    return stars_pool
