import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
        z = np.random.uniform(-10**(-3), 10**(-3), N)
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
            v = 1000 * np.random.rand()
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
        ax2.axis([-0.6, 0.6, -0.6, 0.6])

        plt.savefig("Initial Positions.png")
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

        ax.hist(v, bins = 20, facecolor = pantone, alpha = 0.75, density = True, range = (0, 1000))
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
        x = np.linspace(0, 1000, 100)

        print("Initial temperature: ")
        print(np.round(Temperature, 2))
        ax.plot(x, Ion.Maxwellian(x, Temperature, mass), color = orange, linestyle = 'dashed', linewidth = 3, label = "Maxwellian")

        plt.legend()
        plt.savefig("Initial Speeds.png")
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
        ax1.plot(np.linspace(0, 2*np.pi, 100), (1/(2*np.pi))*np.ones(100), color = orange, linestyle = 'dashed', linewidth = 3, label = "Unifrom")
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
        ax1.legend()
        ax2.legend()

        plt.savefig("Initial Directions.png")
        plt.show()


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
        z (float): z-distance of the particle
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


def EvolveEuler(Ba, BaH, filename, temp):
    """Euler method. During each timestep evolve
    in the quadrupole pseudopotential, add Coulomb
    interactions and then laser cool. Save the
    velocity data to calculate temperatures later.
    
    Args:
        Ba (obj): object containing parameters
        of a Ba+ ion such as position and velocities
    
        BaH (obj): object containing parameters
        of a BaH+ ion such as position and velocities
    
        filename (str): title of the file where the
        evolved velocities are saved to; also used
        to trigger laser cooling
        
        temp (float): initial temperature of
        the simulation, used for writing the 
        file
    """

    Ba_speeds = np.array([])
    BaH_speeds = np.array([])

    for t in range(T):

        # Merge Ba and BaH positions and velocities
        # The data is saved such that the Ba+ data comes first followed by the BaH+ data
        x = np.concatenate((Ba.x, BaH.x))
        y = np.concatenate((Ba.y, BaH.y))
        z = np.concatenate((Ba.z, BaH.z))

        t_reduced = t
        while t_reduced > T_ms:
            t_reduced -= T_ms

        if t_reduced >= T_ms - T_secular:
            Ba_speeds = np.append(Ba_speeds, np.sqrt(Ba.vx**2 + Ba.vy**2 + Ba.vz**2))
            BaH_speeds = np.append(BaH_speeds, np.sqrt(BaH.vx**2 + BaH.vy**2 + BaH.vz**2))

        # Check progress
        if t%500 == 0:
            print("Calculating Euler's method, " + str(np.round((100*t/T), 3)) + "%")

        XCoulombForce, YCoulombForce, ZCoulombForce = CoulombForce(x, y, z, Ba.N + BaH.N)
        XCoulombForceBa, YCoulombForceBa, ZCoulombForceBa = XCoulombForce[:Ba.N:], YCoulombForce[:Ba.N:], ZCoulombForce[:Ba.N:]
        XCoulombForceBaH, YCoulombForceBaH, ZCoulombForceBaH = XCoulombForce[Ba.N::], YCoulombForce[Ba.N::], ZCoulombForce[Ba.N::]

        Ba.vx = Ba.vx + ((XRFForce(Ba.mass, Ba.x, t*dt, a_lc, q_lc) + XCoulombForceBa)/Ba.mass)*dt
        Ba.vy = Ba.vy + ((YRFForce(Ba.mass, Ba.y, t*dt, a_lc, q_lc) + YCoulombForceBa)/Ba.mass)*dt
        Ba.vz = Ba.vz + ((ZPseudoForce(Ba.mass, Ba.z, a_lc, q_lc) + ZCoulombForceBa)/Ba.mass)*dt
        Ba.x = Ba.x + Ba.vx*dt
        Ba.y = Ba.y + Ba.vy*dt
        Ba.z = Ba.z + Ba.vz*dt

        BaH.vx = BaH.vx + ((XRFForce(BaH.mass, BaH.x, t*dt, a_sc, q_sc) + XCoulombForceBaH)/BaH.mass)*dt
        BaH.vy = BaH.vy + ((YRFForce(BaH.mass, BaH.y, t*dt, a_sc, q_sc) + YCoulombForceBaH)/BaH.mass)*dt
        BaH.vz = BaH.vz + ((ZPseudoForce(BaH.mass, BaH.z, a_sc, q_sc) + ZCoulombForceBaH)/BaH.mass)*dt
        BaH.x = BaH.x + BaH.vx*dt
        BaH.y = BaH.y + BaH.vy*dt
        BaH.z = BaH.z + BaH.vz*dt

        # Perform the laser sweep until time-time_end
        if filename == "Laser with sweep" and t < T - int(time_end/dt):
            LaserCool(Ba, float((t/T)*time))
            
        Ba_positions = [Ba.x, Ba.y, Ba.z]
        BaH_positions = [BaH.x, BaH.y, BaH.z]

    with open('Last Secular Period Positions.npy', 'wb') as f:
        np.save(f, Ba_positions)
        np.save(f, BaH_positions)
        
    with open('Last Secular Period Speeds.npy', 'wb') as f:
        np.save(f, Ba_speeds)
        np.save(f, BaH_speeds)


def LaserCool(ion, t):
    """Laser cool the ions using two counterpropagating lasers
    for each Cartesian axis.

    Args:
        ion (obj): object containing positions and velocities
        of ion species to be cooled
    
        t (float): time used for frequency sweep, [s]
    """

    f_0 = LinearFrequencySweep(t)

    for n in range(ion.N):
        if ion.state[n] == "e":
            EmitAPhoton(ion, n)

        # XY Laser
        v_proj = -(ion.vx[n] + ion.vy[n])/np.sqrt(2)
        # Use the Accept-Reject algorithm to decide whether the ion absorbs a photon or not
        lorentzian = Lorentzian(v_proj, f_0)
        randomNumber = (3.12*10**(-8))*np.random.uniform() # 3.12*10**(-8) is approximately the peak of my Lorentzian curve
        # Inelastic collision, photon momentum fully transferred to the ion
        if (randomNumber < lorentzian) and ion.state[n] == "g":
            ion.vx[n] = ion.vx[n] + ((h*f_0)/(np.sqrt(2)*ion.mass*c))
            ion.vy[n] = ion.vy[n] + ((h*f_0)/(np.sqrt(2)*ion.mass*c))
            ion.state[n] = "e"

        # XY Laser
        v_proj = (-ion.vx[n] + ion.vy[n])/np.sqrt(2)
        # Use the Accept-Reject algorithm to decide whether the ion absorbs a photon or not
        lorentzian = Lorentzian(v_proj, f_0)
        randomNumber = (3.12*10**(-8))*np.random.uniform() # 3.12*10**(-8) is approximately the peak of my Lorentzian curve
        # Inelastic collision, photon momentum fully transferred to the ion
        if (randomNumber < lorentzian) and ion.state[n] == "g":
            ion.vx[n] = ion.vx[n] + ((h*f_0)/(np.sqrt(2)*ion.mass*c))
            ion.vy[n] = ion.vy[n] - ((h*f_0)/(np.sqrt(2)*ion.mass*c))
            ion.state[n] = "e"

        # XY Laser
        v_proj = (ion.vx[n] - ion.vy[n])/np.sqrt(2)
        # Use the Accept-Reject algorithm to decide whether the ion absorbs a photon or not
        lorentzian = Lorentzian(v_proj, f_0)
        randomNumber = (3.12*10**(-8))*np.random.uniform() # 3.12*10**(-8) is approximately the peak of my Lorentzian curve
        # Inelastic collision, photon momentum fully transferred to the ion
        if (randomNumber < lorentzian) and ion.state[n] == "g":
            ion.vx[n] = ion.vx[n] - ((h*f_0)/(np.sqrt(2)*ion.mass*c))
            ion.vy[n] = ion.vy[n] + ((h*f_0)/(np.sqrt(2)*ion.mass*c))
            ion.state[n] = "e"

        # XY Laser
        v_proj = (ion.vx[n] + ion.vy[n])/np.sqrt(2)
        # Use the Accept-Reject algorithm to decide whether the ion absorbs a photon or not
        lorentzian = Lorentzian(v_proj, f_0)
        randomNumber = (3.12*10**(-8))*np.random.uniform() # 3.12*10**(-8) is approximately the peak of my Lorentzian curve
        # Inelastic collision, photon momentum fully transferred to the ion
        if (randomNumber < lorentzian) and ion.state[n] == "g":
            ion.vx[n] = ion.vx[n] - ((h*f_0)/(np.sqrt(2)*ion.mass*c))
            ion.vy[n] = ion.vy[n] - ((h*f_0)/(np.sqrt(2)*ion.mass*c))
            ion.state[n] = "e"
            
        # Z Laser
        v_proj = -ion.vz[n]
        # Use the Accept-Reject algorithm to decide whether the ion absorbs a photon or not
        lorentzian = Lorentzian(v_proj, f_0)
        randomNumber = (3.12*10**(-8))*np.random.uniform() # 3.12*10**(-8) is approximately the peak of my Lorentzian curve
        # Inelastic collision, photon momentum fully transferred to the ion
        if (randomNumber < lorentzian) and ion.state[n] == "g":
            ion.vz[n] = ion.vz[n] + ((h*f_0)/(ion.mass*c))
            ion.state[n] = "e"

        # Z Laser
        v_proj = ion.vz[n]
        # Use the Accept-Reject algorithm to decide whether the ion absorbs a photon or not
        lorentzian = Lorentzian(v_proj, f_0)
        randomNumber = (3.12*10**(-8))*np.random.uniform() # 3.12*10**(-8) is approximately the peak of my Lorentzian curve
        # Inelastic collision, photon momentum fully transferred to the ion
        if (randomNumber < lorentzian) and ion.state[n] == "g":
            ion.vz[n] = ion.vz[n] - ((h*f_0)/(ion.mass*c))
            ion.state[n] = "e"

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
            frequencies[t] = LinearFrequencySweep(times[t])

    font = {'fontname':'Cambria'}
    fig, ax = plt.subplots(1, 1, figsize = (5, 4))

    ax.plot((10**3)*times, (10**(-9))*frequencies, 'k-')
    ax.set_xlabel("Time [ms]", **font, size = 18)
    ax.set_ylabel("f [GHz]", **font, size = 18)

    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    plt.savefig("Frequency Sweep.png")
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

    fig, ax = plt.subplots(1, 1, figsize = (5, 4))
    ax.plot((10**(-9))*freq, pdf, 'k-')
    ax.set_xlabel("f [GHz]", **font, size = 18)
    ax.set_ylabel("g [f]", **font, size = 18)
    
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    
    plt.savefig("Lorentzian VS Frequency.png")
    plt.show()


def PlotVelocities(v_Ba, v_BaH):
    
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

    # Turn on the bin edges
    plt.rcParams["patch.force_edgecolor"] = True    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

    x = np.linspace(0, 10, 100)

    ax1.hist(v_Ba, bins = 20, facecolor = pantone, alpha = 0.75, density = True, range = (0, 10))
    #ax1.axvline(np.sqrt(np.mean(v_Ba**2)), color = 'k', linestyle = 'dashed', linewidth = 2, label = "RMS Velocity")

    v_Ba = np.delete(v_Ba, np.argwhere(v_Ba > 500))
    v_Ba_RMS = np.sqrt(np.mean(v_Ba**2))
    T_Ba = m_Ba*v_Ba_RMS**2/(3*k_B)

    print("Ba temperature: ")
    print(np.round(T_Ba, 2))
    ax1.plot(x, Ion.Maxwellian(x, T_Ba, m_Ba), color = orange, linestyle = 'dashed', linewidth = 3, label = "Maxwellian")

    ax1.set_xlabel("Velocity [m/s]", **font, size = 15)
    ax1.set_ylabel("PDF", **font, size = 15)
    ax1.set_title("$Ba^+$", **font, size = 15)
    ax1.grid(axis = 'y')
    #plt.xlim(0, 6)
    #ax1.axis([0, 60, 0, 0.05])
    ax1.tick_params(axis="y", direction="in")
    ax1.tick_params(axis="x", direction="in")
    ax1.legend()

    ax2.hist(v_BaH, bins = 20, facecolor = pantone, alpha = 0.75, density = True, range = (0, 10))
    #ax2.axvline(np.sqrt(np.mean(v_BaH**2)), color = 'k', linestyle = 'dashed', linewidth = 2, label = "RMS Velocity")

    v_BaH = np.delete(v_BaH, np.argwhere(v_BaH > 500))
    v_BaH_RMS = np.sqrt(np.mean(v_BaH**2))
    T_BaH = m_Ba*v_BaH_RMS**2/(3*k_B)
    
    print("BaH temperature: ")
    print(np.round(T_BaH, 2))
    ax2.plot(x, Ion.Maxwellian(x, T_BaH, m_BaH), color = orange, linestyle = 'dashed', linewidth = 3, label = "Maxwellian")

    ax2.set_xlabel("Velocity [m/s]", **font, size = 15)
    ax2.set_ylabel("PDF", **font, size = 15)
    ax2.set_title("$BaH^+$", **font, size = 15)
    ax2.grid(axis = 'y')
    #plt.xlim(0, 6)
    #ax2.axis([0, 60, 0, 0.05])
    ax2.tick_params(axis="y", direction="in")
    ax2.tick_params(axis="x", direction="in")
    ax2.legend()

    plt.savefig("Velocity Distributions over a Secular Period.png")
    plt.show()


def PlotCrystallisation(BaPositions, BaHPositions, temp):
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
    ax1.plot(BaZ, BaX, 'o', markersize = 4, mec = 'k', mfc = pantone, label = '$Ba^+$')
    ax1.plot(BaHZ, BaHX, 'o', markersize = 4, mec = 'k', mfc = 'w', label = '$BaH^+$')

    #XY-plane
    ax2.plot(BaY, BaX, 'o', markersize = 4, mec = 'k', mfc = pantone)
    ax2.plot(BaHY, BaHX, 'o', markersize = 4, mec = 'k', mfc = 'w')

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

    ax1.axis([-1200, 1200, -300, 300])
    ax2.axis([-100, 100, -100, 100])

    ax1.legend(loc = "upper right")
    plt.savefig("Crystallisation " + str(temp) +"K.png")
    plt.show()


def PlotCrossSection(BaPositions, BaHPositions):
    """Plot the XY-positions of ions in the [-20 um, 20 um]
    range to see if the ions have some crystal structure."""

    BaX, BaY, BaZ = (10**6)*BaPositions
    BaHX, BaHY, BaHZ = (10**6)*BaHPositions

    # XY positions of the atoms in the central cross-section of width 10 um
    BaX_Section, BaY_Section = np.array([]), np.array([])
    BaHX_Section, BaHY_Section = np.array([]), np.array([])

    for n in range(N_Ba):
        if BaZ[n] < 20 and BaZ[n] > -20:
            BaX_Section = np.append(BaX_Section, BaX[n])
            BaY_Section = np.append(BaY_Section, BaY[n])

    for n in range(N_BaH):
        if BaHZ[n] < 20 and BaHZ[n] > -20:
            BaHX_Section = np.append(BaHX_Section, BaHX[n])
            BaHY_Section = np.append(BaHY_Section, BaHY[n])

    figure, ax = plt.subplots(1, 1, figsize = (6,6))

    #XY-plane
    ax.plot(BaY_Section, BaX_Section, 'o', markersize = 10, mec = 'k', mfc = pantone, label = '$Ba^+$')
    ax.plot(BaHY_Section, BaHX_Section, 'o', markersize = 10, mec = 'k', mfc = 'w', label = '$BaH^+$')

    ax.set_ylabel("X [$\mu m$]", **font, size = 18)
    ax.set_xlabel("Y [$\mu m$]", **font, size = 18)

    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    ax.axis([-100, 100, -100, 100])

    ax.legend(loc = "upper right")
    plt.savefig("Cross-Section.png")
    plt.show()


def Temperatures(v, N_ions, mass):
    """Calculate the mean temperature for each ms,
    averaged over a secular period. In particular,
    we average the velocity of an ion over the
    secular period and then calculate the RMS
    over all ions. Finally, convert the RMS
    into temperature. """

    temp = np.zeros(ms) # Temperatures of ions for each ms

    for millisecond in range(ms): # Loop over the milliseconds in the simulation
        v_avg = np.zeros(N_ions) # Average velocity over a secular period for each ion
        for timestep in range(T_secular): # Loop over the number of elements saved for each millisecond
            for n in range(N_ions):
                v_avg[n] += v[(millisecond*T_secular + timestep)*N_ions + n]/T_secular # Average over the secular period

        v_RMS = np.sqrt(np.mean(v_avg**2)) # Average over particles
        temp[millisecond] = mass * v_RMS**2 / (3 * k_B)

    return temp


def PlotTemperatureEvolution(temp_Ba, temp_BaH):

    fig, ax = plt.subplots(1, 1, figsize = (6,6))
    timez = np.linspace(1, ms, ms)
    ax.plot(timez, temp_Ba, 'o', mec = 'k', mfc = pantone, label = "$Ba^+$")
    ax.plot(timez, temp_BaH, 'o', mec = 'green', mfc = 'w', label = "$BaH^+$")
    ax.set_xlabel("Time [ms]", **font, size = 15)
    ax.set_ylabel("Temperature [K]", **font, size = 15)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    plt.legend()
    plt.show()


def Compute(temp):
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

    #EvolveEuler(Ba, BaH, "Without laser")
    #EvolveEuler(Ba, BaH, "Laser without sweep")
    EvolveEuler(Ba, BaH, "Laser with sweep", temp)

    del Ba, BaH


def Analyse(name, temp):
    """Calculate and plot the temperature evolution
    for both species and different directions.
    
    Args:
        name (str): ending of the filename to name 
        the temperature evolution plot
    """

    #Velocities(name)

    with open('Last Secular Period Positions.npy', 'rb') as f:
        BaPos = np.load(f)
        BaHPos = np.load(f)
        
    with open('Last Secular Period Speeds.npy', 'rb') as f:
        v_Ba = np.load(f)
        v_BaH = np.load(f)

    temp_Ba = Temperatures(v_Ba, N_Ba, m_Ba)
    temp_BaH = Temperatures(v_BaH, N_BaH, m_BaH)

    PlotCrystallisation(BaPos, BaHPos, temp)
    PlotCrossSection(BaPos, BaHPos)

    # Plot temperature evolution with temperatures averaged over a secular period
    PlotTemperatureEvolution(temp_Ba, temp_BaH)

    # Plot velocity distribution, accumulated over the final secular period
    PlotVelocities(v_Ba, v_BaH)


# Fundamental constants
k_B = 1.381 * 10**(-23)
h = 6.626 * 10**(-34)
permittivity = 8.854 * 10**(-12)
c = 299792458

# Ba+ and BaH+ particle constants
m_Ba = 137.33 * 1.673 * 10**(-27) # [kg], barium mass
m_BaH = 138.33 * 1.673 * 10**(-27) # [kg], barium hydride mass
charge = 1.602*10**(-19) # Ba+ and BaH+ charge

# Optical properties
l_r = 493 * 10**(-9) # [nm], resonance wavelength
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
N_Ba = 20 # number of Ba+ particles
N_BaH = 20 # number of BaH+ particles
particles = N_Ba+N_BaH
R = 10**(-4) # [m], radius of the sphere over which the particles are uniformly distributed
dt = tau # [s], timestep, cannot be larger than 1/f = 2 * 10^(-6) s
time = 7*10**(-3) # [s], total simulation time in s
time_end = 0.1*10**(-3) # [s], time after laser cooling for the system to reach equilibrium
ms = int(time*10**(3)) # [ms], simulation time in ms
T = int(time/dt) # number of timesteps for the whole simulation
T_RF = int(1/(f*dt)) # timesteps in an RF period
T_secular = int(2/(f*dt*np.sqrt(a_sc+((q_sc**2)/2)))) # Number of timesteps in a secular period
T_ms = int(10**(-3)/dt) # Number of timesteps in 1 ms
temp = 10 # [K], initial temperature

# Plot parameters
pantone = (0, 0.2, 0.625) # Main CERN color
orange = (1, 0.4, 0.3) # For fits
font = {'fontname':'Sans-Serif'} # Font of the plot labels and titles
plt.rcParams['figure.dpi'] = 1000 # Resolution of the plots in the console
plt.rcParams['savefig.dpi'] = 1000 # Resolution of the plots when saving
plt.rcParams["patch.force_edgecolor"] = True # Turn on the bin edges

# Main Code
Compute(temp)
Analyse("laser with sweep 10 K", temp)

#PlotLorentzian()
#PlotAcceptRejectLorentzian()