import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from random import shuffle

class Ion():

    def __init__(self, mass, N, temp):
        """Initialise initial positions, velocities, 
        mass, atomic state of an ion ensemble.
        
        Args:
            mass (float): mass of an ion
            
            N (int): number of ions in the ensemble
        """

        self.x, self.y, self.z = self.InitialPositions(N)
        self.vx, self.vy, self.vz = self.InitialVelocities(N, mass, temp)
        self.mass = mass
        self.N = N
        self.state = np.array(["g" for _ in range(N)])


    def InitialPositions(self, N):
        """Generate an array of uniformly
        distributed particle positions over
        a sphere of radius R. 
        
        Args:
            N (int): number of ions in the ensemble
        
        Returns:
            x, y, z (float): arrays with Cartesian
            coordinates of ions
        """

        # Cylindrical polar coords
        phi = np.random.uniform(0, 2*np.pi, N)
        z = np.random.uniform(-1*10**(-3), 1*10**(-3), N)
        r = np.zeros(N)
        for n in range(N):
            r[n] = self.AcceptRejectRadius(R)

        x = r*np.cos(phi)
        y = r*np.sin(phi)

        self.PlotPositions(x, y, z)

        return x, y, z


    def InitialVelocities(self, N, mass, temp):
        """Generate an array of isotropic
        Maxwellian velocities for a given
        temperature. Isotropic means that 
        the velocity directions are
        distributed uniformly over a unit
        sphere.

        Args:
            N (int): number of ions in the ensemble
            
            mass (float): mass of an ion
            
        Returns:
            vx, vy, vz (float): arrays with Cartesian
            velocities of ions
        """

        # Assume velocities to be isotropic over a unit sphere
        phi = np.random.uniform(0, 2*np.pi, N)
        theta = np.zeros(N)
        v = np.zeros(N)

        for n in range(N):
            theta[n] = self.AcceptRejectTheta()
            v[n] = self.AcceptRejectMaxwell(mass, temp)

        vx = v * np.cos(phi) * np.sin(theta)
        vy = v * np.sin(phi) * np.sin(theta)
        vz = v * np.cos(theta)

        self.PlotVelocityDirections(phi, theta)
        self.PlotSpeeds(v, mass)

        return vx, vy, vz


    def Maxwellian(v, T, mass):
        """Normalised Maxwellian velocity distribution
        in 3D.
        
        Args:
            v (float): ion velocity
            
            T (float): temperature of the ion
            ensemble
            
            mass (float): mass of an ion
            
        Returns:
            (float): value of the Maxwellian pdf
        """    

        return (mass/(2*np.pi*k_B*T))**(3/2) * 4*np.pi*v**2 * np.exp(-mass*v**2/(2*k_B*T))


    def AcceptRejectMaxwell(self, mass, temp):
        """The Accept/Reject method replicates a distribution f(x) by following,
        the following algorithm: 1) generate uniformly distributed random numbers,
        x and y; 2) if y < f(x), accept the value, reject otherwise, 3) Repeat.
    
        Args:
            mass (float): mass of an ion
            
        Returns:
            v (float): velocity of an ion which is obtained using the pdf
            of a Maxwellian velocity distribution.
        """

        while True:
            v = 100 * np.random.rand()
            y = Ion.Maxwellian(np.sqrt(2*k_B*temp/mass), temp, mass) * np.random.rand()
            if y < Ion.Maxwellian(v, temp, mass):
                return v
            else:
                continue


    def AcceptRejectTheta(self):
        """The Accept/Reject method replicates a distribution f(x) by following,
        the following algorithm: 1) generate uniformly distributed random numbers,
        x and y; 2) if y < f(x), accept the value, reject otherwise, 3) Repeat.
    
        Returns:
            theta (float): polar angle for the isotropic distribution over
            a unit sphere, range = [0, pi]
        """

        while True:
            theta = np.pi * np.random.rand()
            y = 0.5 * np.random.rand()
            if y < 0.5 * np.sin(theta):
                return theta
            else:
                continue    


    def AcceptRejectRadius(self, R):
        """The Accept/Reject method replicates a distribution f(x) by following,
        the following algorithm: 1) generate uniformly distributed random numbers,
        x and y; 2) if y < f(x), accept the value, reject otherwise, 3) Repeat.

        Args:
            R (float): radius of the sphere over which the ions are uniformly
            distributed
        
        Returns:
            r (float): radial position for initial positions of ions such 
            that they are uniformly distributed over a sphere of radius R,
            range = [0, R]
        """
    
        while True:
            r = R * np.random.rand()
            y = (2/R) * np.random.rand()
            if y < 2 * r/(R**2):
                return r
            else:
                continue


    def PlotPositions(self, x, y, z):     
        """Generate a 3D plot for intial positions
        of ions.
        
        Args:
            x, y, z (float): arrays with Cartesian
            coordinates of ions
        """
        
        X, Y, Z = (10**3)*x, (10**3)*y, (10**3)*z

        figure, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,4), gridspec_kw={'width_ratios': [3, 2]})

        #XZ-plane
        ax1.plot(Z, X, 'o', markersize = 4, mec = 'k', mfc = pantone)

        #XY-plane
        ax2.plot(Y, X, 'o', markersize = 4, mec = 'k', mfc = pantone)

        ax1.set_ylabel("X [$mm$]", **font, size = 18)
        ax1.set_xlabel("Z [$mm$]", **font, size = 18)
        ax2.set_xlabel("Y [$mm$]", **font, size = 18)

        ax1.tick_params(axis="y", direction="in")
        ax1.tick_params(axis="x", direction="in")
        ax2.tick_params(axis="y", direction="in")
        ax2.tick_params(axis="x", direction="in")

        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax2.yaxis.set_ticks_position('both')
        ax2.xaxis.set_ticks_position('both')

        ax1.axis([-1.2, 1.2, -0.6, 0.6])
        ax2.axis([-0.15, 0.15, -0.15, 0.15])

        plt.savefig("Initial Positions.pdf")
        plt.show()


    def PlotSpeeds(self, v, mass):
        """Generate a histogram for intial speeds
        of ions.

        Args:
            v (float): array with ion speeds

            mass (float): mass of an ion
        """
        
        # Turn on the bin edges
        plt.rcParams["patch.force_edgecolor"] = True    

        fig, ax = plt.subplots(1, 1, figsize = (7, 4))

        ax.hist(v, bins = 20, facecolor = pantone, alpha = 0.75, density = True, range = (0, 40))
        ax.set_xlabel("Velocity [m/s]", **font, size = 15)
        ax.set_ylabel("PDF", **font, size = 15)
        #plt.title("Initial speed distribution", **font, size = 15)

        ax.grid(axis = 'y')
        ax.tick_params(axis="y", direction="in")
        ax.tick_params(axis="x", direction="in")
        
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)) # Restrain the y-tics to integers

        # Fit a Maxwellian curve to the velocity distributions
        v_RMS = np.sqrt(np.mean(v**2))
        Temperature = mass*v_RMS**2/(3*k_B)
        x = np.linspace(0, 40, 100)

        print("Initial temperature: ")
        print(np.round(Temperature, 2))
        ax.plot(x, Ion.Maxwellian(x, Temperature, mass), color = orange, linestyle = 'dashed', linewidth = 3, label = "Maxwellian")

        plt.legend(prop={'size': 15})
        plt.savefig("Initial Speeds.pdf")
        plt.show()


    def PlotVelocityDirections(self, phi, theta):
        """Generate histograms for the polar and
        azimuthal angles of the inital ion directions
        
        Args:
            phi (float): array with azimuthal angles, range = [0, 2*pi]
            
            theta (float): array with polar angles, range = [0, pi]
        """

        figure, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,4))

        ax1.hist(phi, bins = 20, color = pantone, alpha = 0.75, density = True)
        ax1.set_xlabel("$\phi$ [rad]", **font, size = 15)
        ax1.set_ylabel("PDF", **font, size = 15)
        #ax1.set_title("Azimuthal angle distribution", **font, size = 15)
        ax1.plot(np.linspace(0, 2*np.pi, 100), (1/(2*np.pi))*np.ones(100), color = orange, linestyle = 'dashed', linewidth = 3, label = "Uniform")
        ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax1.set_xticklabels(['$0$', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'])

        ax2.hist(theta, bins = 20, color = pantone, alpha = 0.75, density = True)
        ax2.set_xlabel("$\Theta$ [rad]", **font, size = 15)
        ax2.set_ylabel("PDF", **font, size = 15)
        #ax2.set_title("Polar angle distribution", **font, size = 15)
        ax2.plot(np.linspace(0, np.pi, 100), 0.5 * np.sin(np.linspace(0, np.pi, 100)), color = orange, linestyle = 'dashed', linewidth = 3, label = "Sine")
        ax2.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax2.set_xticklabels(['$0$', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$'])


        ax1.yaxis.set_major_locator(MaxNLocator(integer=True)) # Restrain the y-tics to integers
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True)) # Restrain the y-tics to integers

        ax1.grid(axis = 'y')
        ax2.grid(axis = 'y')
        ax1.tick_params(axis="y", direction="in")
        ax1.tick_params(axis="x", direction="in")
        ax2.tick_params(axis="y", direction="in")
        ax2.tick_params(axis="x", direction="in")
        ax1.legend(prop={'size': 15})
        ax2.legend(prop={'size': 15})

        plt.savefig("Initial Directions.pdf")
        plt.show()


def PseudoPotential(r, z, a, q, m):
    """Quadrupole pseudoforce due to the oscillating
    RF force.
    
    Args:
        m (float): mass of the particle, [kg]
        
        r (float): radial distance of the particle from
        the centre of the potential well, [m]
    
    Returns:
        (float): restoring force, [N]
    """    

    return (m*(w**2)*(a+((q**2)/2)) * (r**2) / 8) - (m * (w**2) * a * (z**2) / 4)


def RadialPseudoForce(m, r, a, q):
    """Quadrupole pseudoforce due to the oscillating
    RF force.
    
    Args:
        m (float): mass of the particle, [kg]
        
        r (float): radial distance of the particle from
        the centre of the potential well, [m]
    
    Returns:
        (float): restoring force, [N]
    """    

    return -m*(w**2)*(a+((q**2)/2)) * r / 4


def XRFForce(m, x, t, a, q):
    """RF-Force in the x-direction calculated
    from the Mathieu equation for a linear Paul
    trap. 
    
    Args:
        m (float): mass of the particle, [kg]
        
        x (float): x-distance of the particle
        from the centre of the potential well, [m]
        
        t (float): time of the simulation, [s]
        
    Returns:
        (float): force, [N]
    """

    return m * (-a+2*q*np.cos(w*t)) * ((w**2)/4) * x


def YRFForce(m, y, t, a, q):
    """RF-Force in the y-direction calculated
    from the Mathieu equation for a linear Paul
    trap. 
    
    Args:
        m (float): mass of the particle, [kg]
        
        y (float): y-distance of the particle
        from the centre of the potential well, [m]
        
        t (float): time of the simulation, [s]
        
    Returns:
        (float): force, [N]
    """
    

    return m * (-a-2*q*np.cos(w*t)) * ((w**2)/4) * y


def ZPseudoForce(m, z, a, q):
    """Force due to the DC endcap potential.

    Args:
        z (float): axial distance of the particle
        from the centre of the potential well, [m]

        t (float): time of the simulation, [s]

    Returns:
        (float): force, [N]. The parameter A is
        obtained from COMSOL simulations.
    """

    return m * (w**2) * a * z / 2


def CoulombForce(x, y, z, N):
    """Calculate force due to particle-particle
    interactions.
    
    Args:
        x, y, z (float): arrays with cartesian
        coordinates of Ba+ and BaH+ ions
        
        N (int): number of particles
        
    Returns:
        xForce, yForce, zForce (float): arrays with
        forces for N particles, [N].
    """

    xForce = np.zeros(N)    
    yForce = np.zeros(N)
    zForce = np.zeros(N)    

    for j in range(N):
        dist = np.sqrt((x-x[j])**2 + (y-y[j])**2 + (z-z[j])**2)

        xForce += np.where(dist != 0, (1/(4*np.pi*permittivity))*(charge**2)*(x-x[j])/(dist**3), 0)
        yForce += np.where(dist != 0, (1/(4*np.pi*permittivity))*(charge**2)*(y-y[j])/(dist**3), 0)
        zForce += np.where(dist != 0, (1/(4*np.pi*permittivity))*(charge**2)*(z-z[j])/(dist**3), 0)

    return xForce, yForce, zForce


def EvolveEuler(Ba, BaH, mode):
    """Euler method. During each timestep evolve
    ions using Mathieu equations, add Coulomb
    interactions and then laser cool. Save the
    velocities over the last secular period and
    positions at the last time step.

    Args:
        Ba (obj): object containing parameters
        of a Ba+ ion such as position and velocities
    
        BaH (obj): object containing parameters
        of a BaH+ ion such as position and velocities
    
        mode (str): trigger laser cooling
    """

    Ba_speeds = np.array([])
    BaH_speeds = np.array([])

    for t in range(T):

        # Merge Ba and BaH positions and velocities
        # The data is saved such that the Ba+ data comes first followed by the BaH+ data
        x = np.concatenate((Ba.x, BaH.x))
        y = np.concatenate((Ba.y, BaH.y))
        z = np.concatenate((Ba.z, BaH.z))

        if t >= T - T_secular:
            Ba_speeds = np.append(Ba_speeds, np.sqrt(Ba.vx**2 + Ba.vy**2 + Ba.vz**2))
            BaH_speeds = np.append(BaH_speeds, np.sqrt(BaH.vx**2 + BaH.vy**2 + BaH.vz**2))

        # Check progress
        if t%500 == 0:
            print("Calculating Euler's method, " + str(np.round((100*t/T), 3)) + "%")

        XCoulombForce, YCoulombForce, ZCoulombForce = CoulombForce(x, y, z, Ba.N + BaH.N)
        XCoulombForceBa, YCoulombForceBa, ZCoulombForceBa = XCoulombForce[:Ba.N:], YCoulombForce[:Ba.N:], ZCoulombForce[:Ba.N:]
        XCoulombForceBaH, YCoulombForceBaH, ZCoulombForceBaH = XCoulombForce[Ba.N::], YCoulombForce[Ba.N::], ZCoulombForce[Ba.N::]

        Ba.vx = Ba.vx + ((RadialPseudoForce(Ba.mass, Ba.x, a_lc, q_lc) + XCoulombForceBa)/Ba.mass)*dt
        Ba.vy = Ba.vy + ((RadialPseudoForce(Ba.mass, Ba.y, a_lc, q_lc) + YCoulombForceBa)/Ba.mass)*dt
        Ba.vz = Ba.vz + ((ZPseudoForce(Ba.mass, Ba.z, a_lc, q_lc) + ZCoulombForceBa)/Ba.mass)*dt
        Ba.x = Ba.x + Ba.vx*dt
        Ba.y = Ba.y + Ba.vy*dt
        Ba.z = Ba.z + Ba.vz*dt

        BaH.vx = BaH.vx + ((RadialPseudoForce(BaH.mass, BaH.x, a_sc, q_sc) + XCoulombForceBaH)/BaH.mass)*dt
        BaH.vy = BaH.vy + ((RadialPseudoForce(BaH.mass, BaH.y, a_sc, q_sc) + YCoulombForceBaH)/BaH.mass)*dt
        BaH.vz = BaH.vz + ((ZPseudoForce(BaH.mass, BaH.z, a_sc, q_sc) + ZCoulombForceBaH)/BaH.mass)*dt
        BaH.x = BaH.x + BaH.vx*dt
        BaH.y = BaH.y + BaH.vy*dt
        BaH.z = BaH.z + BaH.vz*dt

        # Perform the laser sweep until time-time_end
        if mode == "laser" and t < T - int(time_end/dt):
            LaserCool(Ba, float((t/T)*time))
            
        Ba_positions = [Ba.x, Ba.y, Ba.z]
        BaH_positions = [BaH.x, BaH.y, BaH.z]

    with open('Positions.npy', 'wb') as f:
        np.save(f, Ba_positions)
        np.save(f, BaH_positions)
        
    with open('Speeds.npy', 'wb') as f:
        np.save(f, Ba_speeds)
        np.save(f, BaH_speeds)


def LaserCool(Ba, t):
    """Laser axis is along the y=x=z and y=-x, z=0 directions.

    Args:
        ion (obj): object containing positions and velocities
        of ion species to be cooled
    
        t (float): time used for frequency sweep, [s]
        
        filename (str): title of the file used to trigger
        or avoid frequency sweep
        
        ccd (int): 2D array containing the number of
        scattered photons at certain points in space
    
    """

    f_0 = LinearFrequencySweep(t)

    for n in range(Ba.N):
        if Ba.state[n] == "e":
            EmitAPhoton(Ba, n)

        # Absorb a photon from a randomly selected laser
        lasers = [FirstLaser, SecondLaser, ThirdLaser, FourthLaser, FifthLaser, SixthLaser]
        shuffle(lasers)

        for laser in lasers:
            laser(Ba, n, f_0)


def FirstLaser(Ba, n, f_0):
    
    # Z Laser
    v_proj = -Ba.vz[n]
    # Use the Accept-Reject algorithm to decide whether the ion absorbs a photon or not
    lorentzian = Lorentzian(v_proj, f_0)
    randomNumber = (3.12*10**(-8))*np.random.uniform() # 3.12*10**(-8) is approximately the peak of my Lorentzian curve
    # Inelastic collision, photon momentum fully transferred to the ion
    if (randomNumber < lorentzian) and Ba.state[n] == "g":
        Ba.vz[n] = Ba.vz[n] + ((h*f_0)/(Ba.mass*c))
        Ba.state[n] = "e"


def SecondLaser(Ba, n, f_0):
    
    # Z Laser
    v_proj = Ba.vz[n]
    # Use the Accept-Reject algorithm to decide whether the ion absorbs a photon or not
    lorentzian = Lorentzian(v_proj, f_0)
    randomNumber = (3.12*10**(-8))*np.random.uniform() # 3.12*10**(-8) is approximately the peak of my Lorentzian curve
    # Inelastic collision, photon momentum fully transferred to the ion
    if (randomNumber < lorentzian) and Ba.state[n] == "g":
        Ba.vz[n] = Ba.vz[n] - ((h*f_0)/(Ba.mass*c))
        Ba.state[n] = "e"


def ThirdLaser(Ba, n, f_0):
    
    # XY Laser
    v_proj = (Ba.vx[n] + Ba.vy[n])/np.sqrt(2)
    # Use the Accept-Reject algorithm to decide whether the ion absorbs a photon or not
    lorentzian = Lorentzian(v_proj, f_0)
    randomNumber = (3.12*10**(-8))*np.random.uniform() # 3.12*10**(-8) is approximately the peak of my Lorentzian curve
    # Inelastic collision, photon momentum fully transferred to the ion
    if (randomNumber < lorentzian) and Ba.state[n] == "g":
        Ba.vx[n] = Ba.vx[n] - ((h*f_0)/(np.sqrt(2)*Ba.mass*c))
        Ba.vy[n] = Ba.vy[n] - ((h*f_0)/(np.sqrt(2)*Ba.mass*c))
        Ba.state[n] = "e"


def FourthLaser(Ba, n, f_0):
    
    # XY Laser
    v_proj = -(Ba.vx[n] + Ba.vy[n])/np.sqrt(2)
    # Use the Accept-Reject algorithm to decide whether the ion absorbs a photon or not
    lorentzian = Lorentzian(v_proj, f_0)
    randomNumber = (3.12*10**(-8))*np.random.uniform() # 3.12*10**(-8) is approximately the peak of my Lorentzian curve
    # Inelastic collision, photon momentum fully transferred to the ion
    if (randomNumber < lorentzian) and Ba.state[n] == "g":
        Ba.vx[n] = Ba.vx[n] + ((h*f_0)/(np.sqrt(2)*Ba.mass*c))
        Ba.vy[n] = Ba.vy[n] + ((h*f_0)/(np.sqrt(2)*Ba.mass*c))
        Ba.state[n] = "e"


def FifthLaser(Ba, n, f_0):
    
    # XY Laser
    v_proj = (-Ba.vx[n] + Ba.vy[n])/np.sqrt(2)
    # Use the Accept-Reject algorithm to decide whether the ion absorbs a photon or not
    lorentzian = Lorentzian(v_proj, f_0)
    randomNumber = (3.12*10**(-8))*np.random.uniform() # 3.12*10**(-8) is approximately the peak of my Lorentzian curve
    # Inelastic collision, photon momentum fully transferred to the ion
    if (randomNumber < lorentzian) and Ba.state[n] == "g":
        Ba.vx[n] = Ba.vx[n] + ((h*f_0)/(np.sqrt(2)*Ba.mass*c))
        Ba.vy[n] = Ba.vy[n] - ((h*f_0)/(np.sqrt(2)*Ba.mass*c))
        Ba.state[n] = "e"


def SixthLaser(Ba, n, f_0):

    # XY Laser
    v_proj = (Ba.vx[n] - Ba.vy[n])/np.sqrt(2)
    # Use the Accept-Reject algorithm to decide whether the ion absorbs a photon or not
    lorentzian = Lorentzian(v_proj, f_0)
    randomNumber = (3.12*10**(-8))*np.random.uniform() # 3.12*10**(-8) is approximately the peak of my Lorentzian curve
    # Inelastic collision, photon momentum fully transferred to the ion
    if (randomNumber < lorentzian) and Ba.state[n] == "g":
        Ba.vx[n] = Ba.vx[n] - ((h*f_0)/(np.sqrt(2)*Ba.mass*c))
        Ba.vy[n] = Ba.vy[n] + ((h*f_0)/(np.sqrt(2)*Ba.mass*c))
        Ba.state[n] = "e"


def LinearFrequencySweep(t):
    """Calculate the frequency at a time t
    fo r a linear frequency sweep with a 
    period over the whole simulation.
    
    Args:
        t (float): current time of the simulation, [s]
        
    Returns:
        (float): laser photon frequency at time t, [Hz]
    """
    
    return f_min + (f_max-f_min) * t / (time - time_end)


def EmitAPhoton(ion, n):
    """Emit a photon in an arbitrary direction.
    Aberration not included.
    
    Args:
        ion (obj): object containing positions and velocities
        of ion species to be cooled
        
        n (int): index of the ion that is emitting a photon
    """

    #Pick random emission directions isotropically over a unit sphere
    phi = np.random.uniform(0, 2*np.pi)
    theta = ion.AcceptRejectTheta()

    freq = AcceptRejectLorentzian()

    # The particle will recoil in the opposite direction of the photon emission
    ion.vx[n] -= ((h*freq)/(ion.mass*c))*np.sin(theta)*np.cos(phi)
    ion.vy[n] -= ((h*freq)/(ion.mass*c))*np.sin(theta)*np.sin(phi)
    ion.vz[n] -= ((h*freq)/(ion.mass*c))*np.cos(theta)

    ion.state[n] = "g"


def PlotFrequencySweep():
    """Plot the swept frequency of the
    laser vs time. """    

    times = np.linspace(0, time-time_end, 1000)
    frequencies = np.zeros(len(times))

    for t in range(len(times)):
            frequencies[t] = LinearFrequencySweep(times[t]) - f_r

    font = {'fontname':'Cambria'}
    fig, ax = plt.subplots(1, 1, figsize = (5, 4))

    ax.plot((10**3)*times, (10**(-9))*frequencies, 'k-')
    ax.set_xlabel("Time [ms]", **font, size = 18)
    ax.set_ylabel("$f-f_r$ [GHz]", **font, size = 18)

    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    plt.savefig("Frequency Sweep.pdf")
    plt.show()


def Lorentzian(v_proj, f_laser):
    """Lorentzian describes the probability of a
    photon absorption event given the velocity of
    an ion projected onto the laser axis.
    
    Args:
        v_proj (float): velocity of an ion projected
        onto the laser axis
    
        f_laser (float): frequency of the laser
    """

    return (Gamma/(2*np.pi)) / (((f_r - f_laser - (v_proj/l_r))**2) + ((Gamma/2)**2))


def AcceptRejectLorentzian():
    """The Accept/Reject method replicates a distribution f(x) by following,
    the following algorithm: 1) generate uniformly distributed random numbers,
    x and y; 2) if y < f(x), accept the value, reject otherwise, 3) Repeat. """

    while True:
        freq = 10*Gamma * np.random.rand() + f_r - 5*Gamma
        y = (3.12*10**(-8))*np.random.rand() # 3.12E-8 is the maximum of the Lorentzian
        if y < Lorentzian(0, freq):
            return freq
        else:
            continue


def PlotAcceptRejectLorentzian():

    freq = np.zeros(10000)
    for n in range(10000):
        freq[n] = AcceptRejectLorentzian()

    plt.hist(freq, bins = 50)
    plt.xlabel("Frequencies (Hz)")
    plt.ylabel("Particles")
    plt.title("Velocity distribution")
    plt.show()


def PlotLorentzian():
    """Plot the Lorentzian probability distribution.
    This curve describes the probability of a Ba+ ion
    in an excited state to emit a photon at a certain
    frequency.
    """

    freq = np.linspace(f_r-5*Gamma, f_r+5*Gamma, 1000, dtype = float)
    pdf = np.zeros(len(freq))

    for fr in range(len(freq)):
        pdf[fr] = Lorentzian(0, freq[fr])

    fig, ax = plt.subplots(1, 1, figsize = (7, 5))
    ax.plot((10**(-9))*(freq - f_r), (10**9)*pdf, color = pantone, linewidth = 4)
    ax.set_xlabel("$f - f_r$ [GHz]", **font, size = 21)
    ax.set_ylabel("g(f) [$GHz^{-1}$]", **font, size = 21)
    
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    
    ax.axvline(0, color = orange, linestyle = 'dashed', linewidth = 4)
    
    ax.arrow(2*Gamma*10**(-9), 16, -Gamma*10**(-9), 0, width = 0.5, head_width = 2, head_length = 0.4*Gamma * 10**(-9), color = orange)
    ax.arrow(-2*Gamma*10**(-9), 16, Gamma*10**(-9), 0, width = 0.5, head_width = 2, head_length = 0.4*Gamma * 10**(-9), color = orange)

    ax.text(0.3*Gamma*10**(-9), 30, "$f_r$", **font, fontsize = 22)
    ax.text(1.3*Gamma*10**(-9), 17, "$\Gamma$", **font, fontsize = 22)

    #ax.legend(prop={'size': 15})
    plt.savefig("Lorentzian.pdf")
    plt.show()


def PlotSpeeds(v_Ba, v_BaH):
    
    """Plot velocity distributions during the
    last secular period of the simulation. Fit
    it to the Maxwellian curve and calculate
    the RMS velocity.
    
    Args:
        v_Ba, v_BaH (float): array containing
        Cartesian velocities of barium and
        barium hydride ions for the last
        secular period of the simulation,
        respectively.
    """ 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

    x1 = np.linspace(0, 30, 100)
    x2 = np.linspace(0, 1000, 100)

    ax1.hist(v_Ba, bins = 20, facecolor = pantone, alpha = 0.75, density = True, range = (0, 50))
    #ax1.axvline(np.sqrt(np.mean(v_Ba**2)), color = 'k', linestyle = 'dashed', linewidth = 2, label = "RMS Velocity")

    v_Ba = np.delete(v_Ba, np.argwhere(v_Ba > 50))
    v_Ba_RMS = np.sqrt(np.mean(v_Ba**2))
    T_Ba = m_Ba*v_Ba_RMS**2/(3*k_B)

    print("Ba temperature: ")
    print(np.round(T_Ba, 2))
    #ax1.plot(x1, Ion.Maxwellian(x1, T_Ba, m_Ba), color = orange, linestyle = 'dashed', linewidth = 3, label = "Maxwellian")

    ax1.set_xlabel("Velocity [m/s]", **font, size = 15)
    ax1.set_ylabel("PDF [s/m]", **font, size = 15)
    ax1.set_title("$Ba^+$", **font, size = 15)
    ax1.grid(axis = 'y')
    #plt.xlim(0, 6)
    #ax1.axis([0, 60, 0, 0.05])
    ax1.tick_params(axis="y", direction="in")
    ax1.tick_params(axis="x", direction="in")
    #ax1.legend()

    ax2.hist(v_BaH, bins = 20, facecolor = pantone, alpha = 0.75, density = True, range = (0, 50))
    #ax2.axvline(np.sqrt(np.mean(v_BaH**2)), color = 'k', linestyle = 'dashed', linewidth = 2, label = "RMS Velocity")

    v_BaH = np.delete(v_BaH, np.argwhere(v_BaH > 50))
    v_BaH_RMS = np.sqrt(np.mean(v_BaH**2))
    T_BaH = m_Ba*v_BaH_RMS**2/(3*k_B)

    print("BaH temperature: ")
    print(np.round(T_BaH, 2))
    #ax2.plot(x2, Ion.Maxwellian(x2, T_BaH, m_BaH), color = orange, linestyle = 'dashed', linewidth = 3, label = "Maxwellian")

    ax2.set_xlabel("Velocity [m/s]", **font, size = 15)
    ax2.set_ylabel("PDF [s/m]", **font, size = 15)
    ax2.set_title("$BaH^+$", **font, size = 15)
    ax2.grid(axis = 'y')
    #plt.xlim(0, 6)
    #ax2.axis([0, 60, 0, 0.05])
    ax2.tick_params(axis="y", direction="in")
    ax2.tick_params(axis="x", direction="in")
    #ax2.legend()

    plt.savefig("Final Speeds.pdf")
    plt.show()


def PlotHydrogenTemperatures(v_BaH):
    
    """Plot velocity distributions during the
    last secular period of the simulation. Fit
    it to the Maxwellian curve and calculate
    the RMS velocity.
    
    Args:
        v_Ba, v_BaH (float): array containing
        Cartesian velocities of barium and
        barium hydride ions for the last
        secular period of the simulation,
        respectively.
    """ 

    v_H = v_BaH # velocity of a hydrogen after photodissociation
    KE_H = m_H * v_H**2/2
    temp_H = 2/(3*k_B) * KE_H

    fig, ax = plt.subplots(1, 1, figsize = (7, 5))

    ax.hist((10**3)*temp_H, bins = 20, facecolor = pantone, alpha = 0.75, density = True, range = (0, 100))

    ax.set_xlabel("Temperatures [$mK$]", **font, size = 15)
    ax.set_ylabel("PDF [$1/mK$]", **font, size = 15)
    #ax.set_title("$H$", **font, size = 15)
    ax.grid(axis = 'y')
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    plt.savefig("Hydrogen Temperatures.pdf")

    # Calculate how many hydrogen atoms were below 10 mK
    notsocoldHydrogen = 0
    for n in range(len(temp_H)):
        if temp_H[n] < 10**(-2):
            notsocoldHydrogen += 1/T_secular

    # Calculate how many hydrogen atoms were below 1 mK
    coldHydrogen = 0
    for n in range(len(temp_H)):
        if temp_H[n] < 10**(-3):
            coldHydrogen += 1/T_secular

    # Calculate how many hydrogen atoms were below 100 uK
    ultracoldHydrogen = 0
    for n in range(len(temp_H)):
        if temp_H[n] < 10**(-4):
            ultracoldHydrogen += 1/T_secular

    print("The percentage of hydrogen ions below 10 mK is " + str(np.round(notsocoldHydrogen/5, 2)))
    print("The percentage of hydrogen ions below 1 mK is " + str(np.round(coldHydrogen/5, 2)))
    print("The percentage of hydrogen ions below 100 uK is " + str(np.round(ultracoldHydrogen/5, 2)))

    plt.show()
    
    temp_H *= 10**3
    hot_H = np.argwhere(temp_H > 100)
    temp_H = np.delete(temp_H, hot_H)
    print("The average hydrogen temperature is " + str(np.mean(temp_H)) + " mK")



def PlotPositions(BaPositions, BaHPositions):
    """Plot the positions of Ba+ and BaH+
    ions at the end of the simulation in
    the XZ and XY planes.

    Args:
        BaPositions (float): Positions of the
        Ba+ ions at the end of the simulation

        BaHPositions (float): Positions of the
        BaH+ ions at the end of the simulation
        
        temp (float): initial temperature of
        the simulation, used for reading the 
        file
    """

    BaX, BaY, BaZ = (10**6)*BaPositions
    BaHX, BaHY, BaHZ = (10**6)*BaHPositions

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,4), gridspec_kw={'width_ratios': [3, 2]})

    #XZ-plane
    ax1.plot(BaZ, BaX, 'o', markersize = 10, mec = 'k', mfc = pantone, label = '$Ba^+$')
    ax1.plot(BaHZ, BaHX, 'o', markersize = 10, mec = 'k', mfc = 'w', label = '$BaH^+$')

    #XY-plane
    ax2.plot(BaY, BaX, 'o', markersize = 10, mec = 'k', mfc = pantone)
    ax2.plot(BaHY, BaHX, 'o', markersize = 10, mec = 'k', mfc = 'w')

    # Add a circle separating cooled particles from uncooled
    # radius = (10**6)*np.max(np.abs(np.sqrt(BaPositions[0]**2 + BaPositions[1]**2 + BaPositions[2]**2)))
    # phi = np.linspace(0, 2*np.pi, 100)
    # x = radius * np.cos(phi)
    # y = radius * np.sin(phi)
    # ax1.plot(x, y, 'r-')
    # ax2.plot(x, y, 'r-')

    # Get rid of vertical ticks in the XY-plane plot
    #ax2.tick_params(labelleft = False)

    ax1.set_ylabel("X [$\mu m$]", **font, size = 18)
    ax1.set_xlabel("Z [$\mu m$]", **font, size = 18)
    ax2.set_xlabel("Y [$\mu m$]", **font, size = 18)

    ax1.tick_params(axis="y", direction="in")
    ax1.tick_params(axis="x", direction="in")
    ax2.tick_params(axis="y", direction="in")
    ax2.tick_params(axis="x", direction="in")

    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')

    ax1.axis([-1200, 1200, -600, 600])
    ax2.axis([-100, 100, -100, 100])

    # x1 = np.linspace(-1200, 1200, 5, dtype = int)
    # y1 = np.linspace(-600, 600, 7, dtype = int)

    # ax1.set_yticklabels(y1, fontsize = 15, **font)
    # ax1.set_xticklabels(x1, fontsize = 15, **font)

    # x2 = np.linspace(-100, 100, 5, dtype = int)
    # y2 = np.linspace(-100, 100, 9, dtype = int)

    # ax2.set_yticklabels(y2, fontsize = 15, **font)
    # ax2.set_xticklabels(x2, fontsize = 15, **font)

    ax1.legend(loc = "upper right", prop={'size': 18}, markerscale = 1)
    plt.savefig("Final Positions.pdf")
    plt.show()


def PlotCrossSection(BaPositions, BaHPositions):
    """Plot the XY-positions of ions in the [-20 um, 20 um]
    range to see if the ions have some crystal structure."""

    BaX, BaY, BaZ = (10**6)*BaPositions
    BaHX, BaHY, BaHZ = (10**6)*BaHPositions

    # XY positions of the atoms in the central cross-section of width 10 um
    BaX_Section, BaY_Section, BaZ_Section = np.array([]), np.array([]), np.array([])
    BaHX_Section, BaHY_Section, BaHZ_Section = np.array([]), np.array([]), np.array([])

    for n in range(N_Ba):
        if BaZ[n] < 20 and BaZ[n] > 5:
            BaX_Section = np.append(BaX_Section, BaX[n])
            BaY_Section = np.append(BaY_Section, BaY[n])
            BaZ_Section = np.append(BaZ_Section, BaZ[n])

    for n in range(N_BaH):
        if BaHZ[n] < 20 and BaHZ[n] > 5:
            BaHX_Section = np.append(BaHX_Section, BaHX[n])
            BaHY_Section = np.append(BaHY_Section, BaHY[n])
            BaHZ_Section = np.append(BaHZ_Section, BaHZ[n])

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,7))

    #XY-plane
    ax1.plot(BaZ, BaX, 'o', markersize = 15, mec = 'k', mfc = pantone, alpha = 0.3)
    ax1.plot(BaHZ, BaHX, 'o', markersize = 15, mec = 'k', mfc = 'w', alpha = 0.3)
    ax1.plot(BaZ_Section, BaX_Section, 'o', markersize = 15, mec = 'k', mfc = pantone)
    ax1.plot(BaHZ_Section, BaHX_Section, 'o', markersize = 15, mec = 'k', mfc = 'w')
    ax1.axvline(0, color = 'k', linewidth = 4, label = 'Cross-section')
    ax1.axvline(20, color = 'k', linewidth = 4)

    #x = np.linspace(-100, 100, 9, dtype = int)
    #y = np.linspace(-100, 100, 9, dtype = int)

    #ax.set_yticklabels(y, fontsize = 15, **font)
    #ax.set_xticklabels(x, fontsize = 15, **font)

    ax1.set_ylabel("X [$\mu m$]", **font, size = 25)
    ax1.set_xlabel("Z [$\mu m$]", **font, size = 25)

    ax1.tick_params(axis="y", direction="in")
    ax1.tick_params(axis="x", direction="in")

    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')

    ax1.axis([-100, 100, -100, 100])

    ax1.legend(loc = "upper right", prop={'size': 22})

    #XY-plane
    ax2.plot(BaY_Section, BaX_Section, 'o', markersize = 15, mec = 'k', mfc = pantone, label = '$Ba^+$')
    ax2.plot(BaHY_Section, BaHX_Section, 'o', markersize = 15, mec = 'k', mfc = 'w', label = '$BaH^+$')

    ax2.set_xlabel("Y [$\mu m$]", **font, size = 25)

    ax2.tick_params(axis="y", direction="in")
    ax2.tick_params(axis="x", direction="in")

    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')

    ax2.axis([-100, 100, -100, 100])
    ax2.legend(loc = "upper right", prop={'size': 22})

    plt.savefig("Cross-Section.pdf")
    plt.show()


def LayerSpeeds(positions, velocities, N_ions):
    """Find the speeds of ions at different radial 
    layers of thickness 10 um."""

    # Initiate the temperatures at different layers
    layerSpeeds = np.zeros(layers)

    # Obtain the Cartesian coordinates in um
    X, Y, Z = (10**6) * positions

    # Obtain the radial distances in um
    r = np.sqrt(X**2 + Y**2)

    # Find the RMS speeds at different layers
    for i in range(layers):
        count = 0 # Count the number of ions in a layer to calculate the RMS speed
        for n in range(N_ions):
            for t in range(T_secular):
                if r[n] > i*5 and r[n] < (i + 1)*5:
                    count += 1
                    layerSpeeds[i] += velocities[N_ions*t+n]**2

        layerSpeeds[i] = np.sqrt(layerSpeeds[i]/count)

    return layerSpeeds


def PlotLayerSpeeds(layerSpeedsBa, layerSpeedsBaH):

    fig, ax = plt.subplots(1, 1, figsize = (7,5))
    r = np.linspace(0, 85, layers)
    ax.plot(r, layerSpeedsBa, 'o', markersize = 10, mec = 'k', mfc = pantone, label = "$Ba^+$")
    ax.plot(r, layerSpeedsBaH, 'o', markersize = 10, mec = 'k', mfc = 'w', label = "$BaH^+$")
    ax.set_xlabel("Radial Distance [$\mu m$]", **font, size = 20)
    ax.set_ylabel("RMS Speed [m/s]", **font, size = 20)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    plt.legend(prop={'size': 20})
    plt.savefig("Layer Speeds.pdf")
    plt.show()
    
    print(layerSpeedsBaH[0])
    print(layerSpeedsBa[0])


def CoulombCoupling(Ba_pos, BaH_pos, v_Ba, v_BaH):
    """Find the average potential-to-kinetic energy ratio
    to characterise the strength of coupling. """

    # Average the speeds over the secular period
    v_Ba_secular = np.zeros(N_Ba)
    v_BaH_secular = np.zeros(N_BaH)

    for n in range(N_Ba):
        for t in range(T_secular):
            v_Ba_secular[n] += v_Ba[t*N_Ba + n]/T_secular

    for n in range(N_Ba):
        for t in range(T_secular):
            v_BaH_secular[n] += v_BaH[t*N_BaH + n]/T_secular
    
    # Calculate the kinetic energies and take the average over the ensemble
    KE_Ba = m_Ba*(v_Ba_secular**2)/2
    KE_BaH = m_BaH*(v_BaH_secular**2)/2

    KE = np.concatenate((KE_Ba, KE_BaH))
    KE_avg = np.mean(KE)
    
    # Perform the similar procedure for the potential energies given the positions at the end of the simulation
    PE_Ba = PseudoPotential(np.sqrt(Ba_pos[0]**2 + Ba_pos[1]**2), Ba_pos[2], a_lc, q_lc, m_Ba)
    PE_BaH = PseudoPotential(np.sqrt(BaH_pos[0]**2 + BaH_pos[1]**2), BaH_pos[2], a_sc, q_sc, m_BaH)
    
    PE = np.concatenate((PE_Ba, PE_BaH))
    PE_avg = np.mean(PE)

    # Obtain the Coulomb coupling constant
    coulombCoupling = PE_avg/KE_avg

    return coulombCoupling


def Compute():
    """Compute positions and velocities for three different
    cases with the same initial conditions."""

    global f_min, f_max

    Ba = Ion(m_Ba, N_Ba, temp)
    BaH = Ion(m_BaH, N_BaH, temp)

    #Laser Sweep
    v_max = np.mean(np.sqrt(Ba.vx**2 + Ba.vy**2 + Ba.vz**2)) # [m/s], maximal value of the velocity projection that will absorb laser photons
    v_min = (h*f_r)/(2*m_Ba*c) + (l_r*Gamma/2) # [m/s], recoil limit + 2xHWHM
    f_max = f_r/(1+(v_min/c))
    f_min = f_r/(1+(v_max/c))

    PlotFrequencySweep()

    EvolveEuler(Ba, BaH, "laser")

    del Ba, BaH


def Analyse():
    """Calculate and plot the temperature evolution
    for both species and different directions.

    """

    #Velocities(name)

    with open('Positions.npy', 'rb') as f:
        Ba_pos = np.load(f)
        BaH_pos = np.load(f)
        
    with open('Speeds.npy', 'rb') as f:
        v_Ba = np.load(f)
        v_BaH = np.load(f)

    # Plot final positions
    PlotPositions(Ba_pos, BaH_pos)

    # Plot velocity distribution, accumulated over the final secular period
    PlotSpeeds(v_Ba, v_BaH)

    # Plot hydrogen temperatures after photodissociation
    PlotHydrogenTemperatures(v_BaH)


# Fundamental constants
k_B = 1.381 * 10**(-23)
h = 6.626 * 10**(-34)
permittivity = 8.854 * 10**(-12)
c = 299792458

# Particle constants
amu = 1.66 * 10**(-27) # atomic mass unit
m_Ba = 137.9 * amu # [kg], barium mass
m_BaH = 138.9 * amu # [kg], barium hydride mass
m_H = 1 * amu
charge = 1.602*10**(-19) # Ba+ and BaH+ charge

# Optical properties
l_r = 493.4 * 10**(-9) # [nm], resonance wavelength
f_r = c/l_r # [Hz], resonance frequency
tau = 7.8 * 10**(-9) # [s], lifetime of the excited state
Gamma = 1/(2*np.pi*tau) # [Hz], resonance full width, FWHM of the Lorentzian, 2xerror in resonance frequency

# Trap parameters
a_lc = -0.001 # For laser cooled Ba+
q_lc = 0.4 # For laser cooled Ba+
a_sc = a_lc * (m_Ba/m_BaH) # For sympathetically cooled BaH+
q_sc = q_lc * (m_Ba/m_BaH) # For sympathetically cooled BaH+
alpha = -0.0243 # Aarhus trap parameter
f = 0.5 * 10**6 # [MHz], RF frequency
w = 2*np.pi*f
r_0 = 0.0075 # [m], radius of the trap
U_endcap = a_lc*m_Ba*(w**2)*r_0**2/(4*charge*alpha) # [V], endcap DC voltage
V_RF = q_lc*m_Ba*(w**2)*r_0**2/(2*charge) # [V], RF voltage amplitude

# Simulation parameters
N_Ba = 500 # number of Ba+ particles
N_BaH = 500 # number of BaH+ particles
particles = N_Ba+N_BaH
R = 1*10**(-4) # [m], radius of the cylinder over which the particles are uniformly distributed
dt = tau # [s], timestep, cannot be larger than 1/f = 2 * 10^(-6) s
time = 3*10**(-3) # [s], total simulation time in s
time_end = 0.1*10**(-3) # [s], time after laser cooling for the system to reach equilibrium
ms = int(time*10**(3)) # [ms], simulation time in ms
T = int(time/dt) # number of timesteps for the whole simulation
T_RF = int(1/(f*dt)) # timesteps in an RF period
T_secular = int(2/(f*dt*np.sqrt(a_sc+((q_sc**2)/2)))) # Number of timesteps in a secular period
T_ms = int(10**(-3)/dt) # Number of timesteps in 1 ms
temp = 1 # [K], initial temperature

# Plot parameters
pantone = (0, 0.2, 0.625) # Main CERN color
orange = (1, 0.4, 0.3) # For fits
font = {'fontname':'Sans-Serif'} # Font of the plot labels and titles
plt.rcParams['figure.dpi'] = 1000 # Resolution of the plots in the console
plt.rcParams['savefig.dpi'] = 1000 # Resolution of the plots when saving
plt.rcParams["patch.force_edgecolor"] = True # Turn on the bin edges
plt.rc('ytick', labelsize = 18)
plt.rc('xtick', labelsize = 18)

# How many 10 um layers we are studying
layers = 17

# Main Code
Compute()
Analyse()

#PlotLorentzian()
