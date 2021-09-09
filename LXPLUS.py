import numpy as np

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

    #EvolveEuler(Ba, BaH, "Without laser")
    #EvolveEuler(Ba, BaH, "Laser without sweep")
    EvolveEuler(Ba, BaH, "Laser with sweep", temp)

    del Ba, BaH


# Fundamental constants
k_B = 1.381 * 10**(-23)
h = 6.626 * 10**(-34)
permittivity = 8.854 * 10**(-12)
c = 299792458

# Ba+ and BaH+ particle constants
m_Ba = 137.9 * 1.673 * 10**(-27) # [kg], barium mass
m_BaH = 138.9 * 1.673 * 10**(-27) # [kg], barium hydride mass
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
N_Ba = 2 # number of Ba+ particles
N_BaH = 2 # number of BaH+ particles
particles = N_Ba+N_BaH
R = 10**(-4) # [m], radius of the sphere over which the particles are uniformly distributed
dt = tau # [s], timestep, cannot be larger than 1/f = 2 * 10^(-6) s
time = 0.1*10**(-3) # [s], total simulation time in s
time_end = 0.1*10**(-3) # [s], time after laser cooling for the system to reach equilibrium
ms = int(time*10**(3)) # [ms], simulation time in ms
T = int(time/dt) # number of timesteps for the whole simulation
T_RF = int(1/(f*dt)) # timesteps in an RF period
T_secular = int(2/(f*dt*np.sqrt(a_sc+((q_sc**2)/2)))) # Number of timesteps in a secular period
T_ms = int(10**(-3)/dt) # Number of timesteps in 1 ms
temp = 1 # [K], initial temperature

# Main Code
Compute(temp)