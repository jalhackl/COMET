import itertools
import numpy as np

#timestep = 0.01

dyn_friction_on = 0

partdim = 3


leesedwards=0
boxchperiodiccentered = 1
simplot = 0

#kt = 1.38e-23
kt = 1

#arbitrarily chosen constants for the potential
hconst = 1
hconst2 = 0

#turn harmonic potential on (1) or off (0)
#harmpot_on=1

#turn fene potential on (1) or off (0)
#fenepot_on=0

#turn quartic potential on (1) or off (0)
quarticpot_on=0

#turn thermostat on (1) or off (0)
thermostat_on = 1

# space-dependent friction
dyn_friction_on = 0

#constants for friction term 
frica = 0.5
fricb = 5

feneK = hconst

#for harm potential
group_movement = False
polymer_movement = True

#set interaction_partners
# group determines which index
interaction_partners = []

#interaction_partner_calc = False

#set D
D=0.001


def random_3d_vector_with_magnitude(magnitude):

    random_vector = np.random.randn(partdim)
    
    # Normalize 
    unit_vector = random_vector / np.linalg.norm(random_vector)
    
    # Scale 
    scaled_vector = unit_vector * magnitude
    
    return scaled_vector

class Particle():
    newid = itertools.count().__next__
    def __init__(self, position, velocity, mass, friction , force , temperature, particle_type = None, interaction_partners = [], interaction_group = 0, group=0, entity_group=0, move_group=0, full_group=0, number=0, central_particle=False, central_particle_group=0, identity=0,
                #const_vel=np.array([1,0,0]),
                active_velocity0 = 1, active_velocity_vec=np.array([1,1,1]), sab=False, eab=False):
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.friction = friction
        self.force = force
        self.temperature = temperature

        #always false
        self.lan = False
        
        self.positions = []
        self.velocities = []
        self.forces = []
        self.kinetic1 = []
        self.potentiale = []
        self.frictions = []


        self.active_velocity0 = active_velocity0
        self.active_velocity_vec = active_velocity_vec

        # set active velocity
        self.active_velocity_vec = active_velocity_vec / np.linalg.norm(active_velocity_vec)
            #active_velocity is given by v0*ek
        self.active_velocity = self.active_velocity0 * self.active_velocity_vec

        self.particle_type = particle_type
        if self.particle_type == "const":
            self.const_vel = random_3d_vector_with_magnitude(self.active_velocity0)


        self.active_velocities = []
        self.active_velocities_vec = []
        self.total_velocities = []

        self.sab = sab
        self.eab = eab

        # new for groups
        self.central_particle = central_particle
        self.central_particle_group = central_particle_group
        self.identity = identity

        

        #new for direct groups
        self.interaction_partners = interaction_partners
        self.interaction_group = interaction_group
        
        
        self.id = Particle.newid()
        

        #counting of steps performed
        self.counter = 0

        self.group = group
        self.entity_group = entity_group
        self.move_group = move_group
        self.full_group = full_group

        self.number = number

        # list to use for clustering
        self.curr_group_list = []


def boxcheck():
    bounce_dims = [0, 1, 2]  # List of dimensions where bounce-back applies (e.g., 1 for Y-axis)
    no_slip_walls = []  # List of dimensions where no-slip boundary conditions apply (e.g., 0 for X-axis, 2 for Z-axis)
    
    if leesedwards == 1:
        if boxchperiodiccentered == 1:
            global layerdisplacement
            
            for Particle in particles:
                for i in range(partdim):
                    while (Particle.position[i] >= boxl * 0.5):
                        Particle.position[i] -= boxl
                    while (Particle.position[i] < -boxl * 0.5):
                        Particle.position[i] += boxl 
        
        if boxchsimple == 1:
            for Particle in particles:
                for i in range(partdim):
                    if ((Particle.position[i] > box[i][1]) or (Particle.position[i] < box[1][0])):
                        Particle.position = Particle.position % boxhl
                        continue
        
        if boxchflow == 1:        
            for Particle in particles:
                for i in range(partdim):
                    if ((Particle.position[i] > box[i][1])):
                        evel = Particle.velocity / np.linalg.norm(Particle.velocity)
                        maxf = min(
                            (boxhl + Particle.position[0]) / evel[0],
                            (boxhl + Particle.position[1]) / evel[1],
                            (boxhl + Particle.position[2]) / evel[2]
                        )
                        Particle.position -= (maxf - 0.0001) * evel
                    if (Particle.position[i] < box[1][0]):
                        evel = Particle.velocity / np.linalg.norm(Particle.velocity)
                        maxf = min(
                            (boxhl + Particle.position[0]) / evel[0],
                            (boxhl + Particle.position[1]) / evel[1],
                            (boxhl + Particle.position[2]) / evel[2]
                        )
                        Particle.position -= (maxf - 0.0001) * evel
    else:
        if boxchperiodiccentered == 1:
            for Particle in particles:
                for i in range(partdim):
                    if i in bounce_dims:  # Apply bounce-back rule
                        if Particle.position[i] >= boxl * 0.5:
                            Particle.position[i] = boxl * 0.5
                            Particle.velocity[i] *= -1  # Reverse velocity

                            if Particle.particle_type == "const":
                                Particle.const_vel *= -1
                            #if Particle.particle_type == "const" or Particle.particle_type == "sab" or Particle.particle_type == "sab_follow":
                            if Particle.particle_type == "const" or Particle.particle_type == "sab":
                                Particle.active_velocity *= -1
                                Particle.active_velocity_vec *= -1
                        elif Particle.position[i] < -boxl * 0.5:
                            Particle.position[i] = -boxl * 0.5
                            Particle.velocity[i] *= -1  # Reverse velocity
                        
                            if Particle.particle_type == "const":
                                Particle.const_vel *= -1
                            #if Particle.particle_type == "const" or Particle.particle_type == "sab" or Particle.particle_type == "sab_follow":
                            if Particle.particle_type == "const" or Particle.particle_type == "sab":
                                Particle.active_velocity *= -1
                                Particle.active_velocity_vec *= -1


                    elif i in no_slip_walls:  # Apply no-slip boundary condition
                        if Particle.position[i] >= boxl * 0.5:
                            Particle.position[i] = boxl * 0.5
                            Particle.velocity[i] = 0  # Set velocity to zero

                            if Particle.particle_type == "const":
                                Particle.const_vel *= -1
                            if Particle.particle_type == "const" or Particle.particle_type == "sab" or Particle.particle_type == "sab_follow":
                                Particle.active_velocity *= -1
                                Particle.active_velocity_vec *= -1
                            

                        elif Particle.position[i] < -boxl * 0.5:
                            Particle.position[i] = -boxl * 0.5
                            Particle.velocity[i] = 0  # Set velocity to zero

                            if Particle.particle_type == "const":
                                Particle.const_vel *= -1
                            if Particle.particle_type == "const" or Particle.particle_type == "sab" or Particle.particle_type == "sab_follow":
                                Particle.active_velocity *= -1
                                Particle.active_velocity_vec *= -1

                    else:  # Apply periodic boundary condition
                        while (Particle.position[i] >= boxl * 0.5):
                            Particle.position[i] -= boxl
                            Particle.boxcross[i] += boxl
                        while (Particle.position[i] < -boxl * 0.5):
                            Particle.position[i] += boxl
                            Particle.boxcross[i] -= boxl


def cart_to_e_unit(evelvec):
    ex = np.array([1,0,0])
    ey = np.array([0,1,0])
    ez = np.array([0,0,1])
    
    x = evelvec[0]
    y = evelvec[1]
    z = evelvec[2]
    
    r = np.linalg.norm(evelvec)
    theta = np.arccos(z/r)
    phi = np.arctan(y/x)
    

    #runit = (np.sin(theta)*np.cos(phi)*ex)+((np.sin(theta)*np.sin(phi))*ey)+(np.cos(theta)*ez)
    
    thetaunit = (np.cos(theta)*np.cos(phi)*ex)+((np.cos(theta)*np.sin(phi))*ey)-(np.sin(theta)*ez)
    phiunit = -(np.sin(phi)*ex)+np.cos(phi)*ey
    
    return thetaunit, phiunit


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

#SABPO (and EABPO)
def integrate_evel_vector(Particle, evelvec):
    if partdim == 3:
        
        randomfactor = np.sqrt(2*D*timestep)
        
        theta = np.random.randn(partdim)*randomfactor
        phi = np.random.randn(partdim)*randomfactor

        
        etheta, ephi = cart_to_e_unit(evelvec)
        
        eestimate = evelvec + np.multiply(etheta , theta) + np.multiply(ephi , phi) - np.multiply( evelvec , (2 * timestep * D ) )
        
        evelvecnew = eestimate / np.linalg.norm(eestimate)
        #alternatively, should be the same for small D and delta t (it is)
        #evelvecnew2 = evelvec + etheta * theta + ephi * phi - 0.5 * (((theta**2) + (phi**2))*evelvec)

        return evelvecnew
    
    elif partdim == 2:

        
        randomfactor = np.sqrt(2*D*timestep)
        
        theta = np.random.randn(partdim)*randomfactor
        phi = np.random.randn(partdim)*randomfactor
        #espherical = coordinates.cartesian_to_spherical(evelvec[0], evelvec[1], evelvec[2]) 
        
        etheta, ephi = cart2pol(evelvec[0], evelvec[1])
        
        #latitude
        #etheta = espherical[1].value
        etheta = 0
        
        #ephi = espherical[2].value
        
        eestimate = evelvec + etheta * theta + ephi * phi
        
        evelvecnew = eestimate / np.linalg.norm(eestimate)
        
        
        #alternatively, should be the same for small D and delta t (it is)
        #evelvecnew2 = evelvec + etheta * theta + ephi * phi - 0.5 * (((theta**2) + (phi**2))*evelvec)
        
            
        return evelvecnew

    else:
        raise ValueError("Wrong dimension for active velocity vector!")


#SABPO (and EABPO)
def integrate_evel_vector_no_particle_argument(evelvec):
    if partdim == 3:
        
        randomfactor = np.sqrt(2*D*timestep)
        
        theta = np.random.randn(partdim)*randomfactor
        phi = np.random.randn(partdim)*randomfactor

        
        etheta, ephi = cart_to_e_unit(evelvec)
        
        eestimate = evelvec + np.multiply(etheta , theta) + np.multiply(ephi , phi) - np.multiply( evelvec , (2 * timestep * D ) )
        
        evelvecnew = eestimate / np.linalg.norm(eestimate)
        #alternatively, should be the same for small D and delta t (it is)
        #evelvecnew2 = evelvec + etheta * theta + ephi * phi - 0.5 * (((theta**2) + (phi**2))*evelvec)

        return evelvecnew
    
    elif partdim == 2:

        
        randomfactor = np.sqrt(2*D*timestep)
        
        theta = np.random.randn(partdim)*randomfactor
        phi = np.random.randn(partdim)*randomfactor
        #espherical = coordinates.cartesian_to_spherical(evelvec[0], evelvec[1], evelvec[2]) 
        
        etheta, ephi = cart2pol(evelvec[0], evelvec[1])
        
        #latitude
        #etheta = espherical[1].value
        etheta = 0
        
        #ephi = espherical[2].value
        
        eestimate = evelvec + etheta * theta + ephi * phi
        
        evelvecnew = eestimate / np.linalg.norm(eestimate)
        
        
        #alternatively, should be the same for small D and delta t (it is)
        #evelvecnew2 = evelvec + etheta * theta + ephi * phi - 0.5 * (((theta**2) + (phi**2))*evelvec)
        
            
        return evelvecnew

    else:
        raise ValueError("Wrong dimension for active velocity vector!")
    
#standard velocity verlet scheme: new positions
def new_pos(Particle):
    
       
    Particle.positions.append(Particle.position) 
    Particle.velocities.append(Particle.velocity) 

    Particle.position = Particle.position + Particle.velocity*timestep + (Particle.force )*(timestep**2)*0.5 / Particle.mass


    #SABPO
    if Particle.sab == True and Particle.lan == False:
        Particle.position = Particle.position + timestep*Particle.active_velocity
        
        
        
    if Particle.eab == True:

        Particle.position = Particle.position + (Particle.active_forces[-1] )*(timestep**2)*0.5 / Particle.mass




#standard velocity verlet scheme: new velocities
def new_vel(Particle):

    actforces = new_forces_thermostat(Particle)  

    Particle.velocity = Particle.velocity + timestep * (actforces + Particle.force) / (2*Particle.mass)
    
    Particle.force = actforces


#standard velocity verlet scheme: new velocities
def new_vel_partners(Particle):

    actforces = new_forces_thermostat_partners(Particle)  

    Particle.velocity = Particle.velocity + timestep * (actforces + Particle.force) / (2*Particle.mass)
    
    Particle.force = actforces


    #EABPO: backflow force on non-active polymer particles
    '''
    if (EABPO == True and Particle.eab == True):
        Particle.velocity = Particle.velocity + timestep * (Particle.active_force + Particle.active_forces[-1]) / (2*Particle.mass)
            #velocity verlet
        #print("compare the forces")
        #print(Particle.active_force)
        #print(Particle.active_forces[-1])
    
    if (EABPO == True):
        if ((Particle.eab == True and EABPO_mono_active_backflow == True) or (Particle.eab == False and EABPO_mono_backflow == True)):
            Particle.velocity = Particle.velocity - timestep * ( (Particle.mass / massparts) * ( total_active_force + old_total_active_force ) ) / (2*Particle.mass)
            #velocity verlet
            #print("compare the TOTAL forces")
            #print(total_active_force)
            #print(old_total_active_force )
    '''

    #SABPO
    if (Particle.sab == True):
        
        
        Particle.active_velocities_vec.append(Particle.active_velocity_vec)
        
        
        if Particle.lan == False:
            new_evel = integrate_evel_vector(Particle, Particle.active_velocity_vec)
            new_active_velocity = Particle.active_velocity0 * new_evel

            Particle.active_velocity_vec = new_evel

            #Particle.active_velocity_vec.append(new_evel)
        else:
            new_temp = new_forces_thermostat_lan(Particle) 
            
            
            
            new_active_velocity = Particle.active_velocity + timestep * (new_temp + Particle.active_force)  / (2*Particle.mass)
            
            Particle.active_force = new_temp
            
            
            Particle.active_forces.append(new_temp)

        Particle.active_velocity = new_active_velocity
        

    if Particle.particle_type == "const":
        if const_vel_try == True:
            Particle.velocity = Particle.const_vel
            Particle.force = np.array([0 for _ in range(partdim)])


def update_active_velocities(active_vels):
    for ia, active_vel in enumerate(active_vels):
        active_vels[ia] = integrate_evel_vector_no_particle_argument(active_vel)




def invert_active_velocities(active_vels):
    for ia, active_vel in enumerate(active_vels):
        active_vels[ia] = active_vel * -1


def new_vel_partners_global_velocities(Particle):


    actforces = new_forces_thermostat_partners(Particle) 

    Particle.velocity = Particle.velocity + timestep * (actforces + Particle.force) / (2*Particle.mass)
    
    Particle.force = actforces


    #EABPO: backflow force on non-active polymer particles
    '''
    if (EABPO == True and Particle.eab == True):
        Particle.velocity = Particle.velocity + timestep * (Particle.active_force + Particle.active_forces[-1]) / (2*Particle.mass)
            #velocity verlet
        #print("compare the forces")
        #print(Particle.active_force)
        #print(Particle.active_forces[-1])
    
    if (EABPO == True):
        if ((Particle.eab == True and EABPO_mono_active_backflow == True) or (Particle.eab == False and EABPO_mono_backflow == True)):
            Particle.velocity = Particle.velocity - timestep * ( (Particle.mass / massparts) * ( total_active_force + old_total_active_force ) ) / (2*Particle.mass)
            #velocity verlet
            #print("compare the TOTAL forces")
            #print(total_active_force)
            #print(old_total_active_force )
    '''

    #SABPO
    if (Particle.sab == True):
        
        
        Particle.active_velocities_vec.append(Particle.active_velocity_vec)
        
        
        if Particle.lan == False:
            new_evel = act_velocity_vectors[Particle.group]
            new_magn = act_velocity_vectors_magnitudes[Particle.group]
            # set to magnitude
            new_active_velocity = new_magn * new_evel

            Particle.active_velocity_vec = new_evel


        Particle.active_velocity = new_active_velocity

    if Particle.particle_type == "const":
        if const_vel_try == True:
            Particle.velocity = Particle.const_vel
            Particle.force = np.array([0 for _ in range(partdim)])


additional_vels = []
def compute_group_vel_additions(particles, entities, abs_vel=None):
    global additional_vels

    additional_vels = []
    if not abs_vel:
        all_vels = []
        for ParticleP in particles:
            all_vels.append(ParticleP.velocity)
        avg_vel = np.mean(all_vels)
        abs_vel = np.abs(avg_vel)

    
    for i in range(entities):
        
        random_vector = random_vector_with_length(abs_vel)
        additional_vels.append(random_vector)



group_temp_vectors = []
def compute_group_temps(entities):
    global group_temp_vectors

    group_temp_vectors = []
    
    for i in range(entities):
        
        random_vector = np.random.randn(partdim)
        group_temp_vectors.append(random_vector)


    

#updating of the acting forces
def new_forces_thermostat_partners(Particle, min_group=1):
    force = 0
    force = force - Particle.friction * Particle.velocity * Particle.mass
    if (harmpot_on == 1):
        hp = harm_potential_partners(Particle)
        force = force - hp
        
    if (fenepot_on == 1):
        fp = fene_potential(Particle)
        force = force - fp
       # print("fenep")
       # print(force)
    if (quarticpot_on == 1):
        qp = quartic_potential(Particle)
        force = force - qp
    
    if not thermostat_for_group: 
        force = force + np.random.randn(partdim)*sigma_thermostat(Particle)
    else:

        force = force + group_temp_vectors[Particle.move_group -  min_group]*sigma_thermostat(Particle)
    
    Particle.forces.append(Particle.force)
   
    return force




#interaction_partner_calc



#Langevin thermostat
#see https://www.vasp.at/wiki/index.php/Langevin_thermostat
#this function computes the variance of the random number used for the random force in the Langevin equation
#via the 'temperature' attribute the temperature can be adjusted for each particle
def sigma_thermostat(Particle):
    if (thermostat_on == 1):
        sigma = np.sqrt(2*kt*Particle.temperature*Particle.friction*Particle.mass/timestep)
    else:
        sigma = 0
    return sigma

only_central_interactions = True
    


#the potential between two particles
def harm_potential(Particle, group_movement=False, polymer_movement = True):
    pot = 0
    pote = 0
    for ParticleP in particles:
        
        if group_movement:
            if Particle.group == ParticleP.group:
                continue

        if polymer_movement:
                    
            if ParticleP.entity_group != Particle.entity_group:
                continue

            if (ParticleP.number != Particle.number+1) and (ParticleP.number != Particle.number-1):
                continue


        if (id(Particle) == id(ParticleP)):
            
            continue
        else:          
            dst = ParticleP.position-Particle.position
           
            vk = dst / np.linalg.norm(dst)
            
            pott = hconst * 2 * np.linalg.norm(dst - hconst2)
            pot = pot - pott * vk
     
            pot = pot / 2

    return pot

#the potential between two particles
def harm_potential_partners(Particle):
    pot = 0
    pote = 0
    interaction_list = Particle.interaction_partners
    central = Particle.central_particle

    if not central:
        for ParticleP in particles:
            if ParticleP.identity in interaction_list:

                if only_central_interactions:
                    if not ParticleP.central_particle:
                        
                        continue

                #else:          
                dst = ParticleP.position-Particle.position
            
                vk = dst / np.linalg.norm(dst)
                
                pott = hconst * 2 * np.linalg.norm(dst - hconst2)
                pot = pot - pott * vk
        
                pot = pot / 2

    return pot


def adopt_central_velocity(Particle):
    interaction_list = Particle.interaction_partners
    central = Particle.central_particle

    if not central:
        for ParticleP in particles:
            if ParticleP.identity in interaction_list:
                if ParticleP.central_particle:
                    if replace_vel:
                        Particle.velocity = ParticleP.velocity
                    else:
                        Particle.velocity = Particle.velocity + ParticleP.velocity
                        
                       
                    break


def adopt_central_active_velocity(Particle, update_active=False):
    interaction_list = Particle.interaction_partners
    central = Particle.central_particle

    if not central:
        for ParticleP in particles:
            if ParticleP.identity in interaction_list:
                if ParticleP.central_particle:
                    if replace_vel:
                        if update_active:
                            Particle.velocity = ParticleP.active_velocity
                        else:
                            Particle.active_velocity = ParticleP.active_velocity
                    else:
                        if update_active:
                            Particle.velocity = Particle.velocity + ParticleP.active_velocity
                        else:
                            Particle.active_velocity = Particle.active_velocity + ParticleP.active_velocity
                    break


#fene potential
def fene_potential(Particle):
    pot = 0

    
    K = feneK
    R0 = 1.5
    epsilon = 1
    sigma_fene = 1
    cutoff = 2 **(1/6)


    K = feneK
    R0 = 1.5
    epsilon = 1
    sigma_fene = 1
    cutoff = 2 **(1/6)


    #fene potential (https://lammps.sandia.gov/doc/bond_fene.html#examples)
    # E = -0.5 * K * R0^2 * ln(1-(r/R0)^2)) + LJ + epsilon
    # -> force:
    # F = K * r / (1 - (r / R0)^2) + (24 * epsilon / r) * (2 (sigma_fene / r)^12 - (sigma_fene / r)^6)
    
    for ParticleP in particles:
        if (id(Particle) == id(ParticleP)):
            
            continue


        
        if ParticleP.entity_group != Particle.entity_group:
            continue

        if (ParticleP.number != Particle.number+1) and (ParticleP.number != Particle.number-1):
            continue

        else: 
            dst = ParticleP.position-Particle.position
            r = np.linalg.norm(dst)
            
            vk = dst / np.linalg.norm(dst)
            if (r < cutoff):
                pott = K * (1 / (1 - (r / R0)**2)) - (24 * epsilon / r) * (2 * (sigma_fene / r)**12 - (sigma_fene / r)**6) 
            else:
                pott = K * (1 / (1 - (r / R0)**2)) 
           
            pot = pot - pott * vk 
   
    return pot


#fene potential
def fene_potential_partners(Particle):
    pot = 0

    interaction_list = Particle.interaction_partners
    central = Particle.central_particle
    
    K = feneK
    R0 = 1.5
    epsilon = 1
    sigma_fene = 1
    cutoff = 2 **(1/6)


    K = feneK
    R0 = 1.5
    epsilon = 1
    sigma_fene = 1
    cutoff = 2 **(1/6)


    #fene potential (https://lammps.sandia.gov/doc/bond_fene.html#examples)
    # E = -0.5 * K * R0^2 * ln(1-(r/R0)^2)) + LJ + epsilon
    # -> force:
    # F = K * r / (1 - (r / R0)^2) + (24 * epsilon / r) * (2 (sigma_fene / r)^12 - (sigma_fene / r)^6)
    
    for ParticleP in particles:
        if (id(Particle) == id(ParticleP)):
            
            continue


        if not central:
            for ParticleP in particles:
                if ParticleP.identity in interaction_list:

                    if only_central_interactions:
                        if not ParticleP.central_particle:
                            continue

                dst = ParticleP.position-Particle.position
                r = np.linalg.norm(dst)
                
                vk = dst / np.linalg.norm(dst)
                if (r < cutoff):
                    pott = K * (1 / (1 - (r / R0)**2)) - (24 * epsilon / r) * (2 * (sigma_fene / r)**12 - (sigma_fene / r)**6) 
                else:
                    pott = K * (1 / (1 - (r / R0)**2)) 
            
                pot = pot - pott * vk 
    
    return pot






def run_simulation(particles, partners=False, move_groups=None, group_additions = False, min_group=1):
    
    # loop for 2 (or more) particles  
    for step in range(steps):
        
        #for thermo
        if thermostat_for_group:
            compute_group_temps(move_groups)

        for Particle in particles:        
            # new positions
            new_pos(Particle)
        for Particle in particles:  
            # new velocities
            
            if partners:

                if Particle.particle_type == "const":
                    
                    new_vel_partners(Particle)
            else:
                new_vel(Particle)

            if adopt_vel:
                adopt_central_velocity(Particle)
            if adopt_act_vel:
                adopt_central_active_velocity(Particle)


        if box:
            boxcheck()

        if group_additions:
            compute_group_vel_additions(particles, move_groups, abs_vel=group_velocity)
            for Particle in particles:  
                curr_group = Particle.move_group

                Particle.velocity = Particle.velocity + additional_vels[curr_group-min_group] 



def permute_groups(particles, main_particle="sab", follow_particle="sab_follow"):
    # Step 1: Organize particles by their original groups
    group_dict = {}
    for particle in particles:
        if particle.group not in group_dict:
            group_dict[particle.group] = []
        group_dict[particle.group].append(particle.identity)  # Store only IDs

    # Step 2: Shuffle the identities and assign to new groups
    all_ids = [p.identity for p in particles]
    random.shuffle(all_ids)  # Shuffle particle identities
    
    # Step 3: Assign shuffled IDs to new groups while keeping original sizes
    new_groups = {}  # Maps new group ID -> list of particle IDs
    start_idx = 0
    for old_group, members in group_dict.items():
        group_size = len(members)  # Keep group sizes identical
        new_groups[old_group] = all_ids[start_idx:start_idx + group_size]
        start_idx += group_size

    # Step 4: Update each particle's attributes
    id_to_particle = {p.identity: p for p in particles}  # Quick lookup by identity
    
    for new_group, new_members in new_groups.items():
        new_central_atom = random.choice(new_members)  # Pick a new central atom

        for particle_id in new_members:
            particle = id_to_particle[particle_id]
            particle.group = new_group  # Assign new group
            particle.interaction_partners = new_members.copy()  # Update partners
            particle.central_particle = new_central_atom  # Assig

            if particle.identity == particle.central_particle:
                particle.particle_type = main_particle
                
            else:
                particle.particle_type = follow_particle

    return particles

def run_simulation_sab_follow(particles, move_groups=None, group_additions = False, min_group=1):
    
    # loop for 2 (or more) particles  
    for step in range(steps):


        
        #for thermo
        if thermostat_for_group:
            compute_group_temps(move_groups)


        for Particle in particles:    

            Particle.curr_group_list.append(Particle.group)
            # new positions
            new_pos(Particle)


        if step > 0 and step % change_groups_after_steps == 0:
            print("change groups")
            particles = permute_groups(particles)
            #for Particle in particles:
            #    interaction_list = Particle.interaction_partners
            #    central = Particle.central_particle
                

        for Particle in particles:  
            # new velocities
            
            #if Particle.particle_type == "const":
            new_vel_partners(Particle)


            #if adopt_vel:
            #    adopt_central_velocity(Particle)
            #if adopt_act_vel:
            adopt_central_active_velocity(Particle)



        if box:
            boxcheck()
        '''
        if group_additions:
            compute_group_vel_additions(particles, move_groups, abs_vel=group_velocity)
            for Particle in particles:  
                curr_group = Particle.move_group

                Particle.velocity = Particle.velocity + additional_vels[curr_group-min_group] 
        '''
        for Particle in particles:
            Particle.active_velocities.append(Particle.active_velocity)
            Particle.total_velocities.append(Particle.velocity + Particle.active_velocity)




def run_simulation_sab_follow_global_vels(particles, move_groups=None, group_additions = False, min_group=1):
    
    # loop for 2 (or more) particles  
    for step in range(steps):
    

        #for thermo
        if thermostat_for_group:
            compute_group_temps(move_groups)


        for Particle in particles:    

            Particle.curr_group_list.append(Particle.group)
            # new positions
            new_pos(Particle)


        if step > 0 and step % change_groups_after_steps == 0:
            print("change groups")
            particles = permute_groups(particles)
            #for Particle in particles:
            #    interaction_list = Particle.interaction_partners
            #    central = Particle.central_particle
                
        if update_global_activity_vector:
            update_active_velocities(act_velocity_vectors)
            if invert_velocity_after_steps:
                if step > 0 and step % invert_velocity_after_steps == 0:
                    invert_active_velocities(act_velocity_vectors)

        
        for Particle in particles:  
            # new velocities

            new_vel_partners_global_velocities(Particle)


        if box:
            boxcheck()
        '''
        if group_additions:
            compute_group_vel_additions(particles, move_groups, abs_vel=group_velocity)
            for Particle in particles:  
                curr_group = Particle.move_group

                Particle.velocity = Particle.velocity + additional_vels[curr_group-min_group] 
        '''
        for Particle in particles:
            Particle.active_velocities.append(Particle.active_velocity)
            Particle.total_velocities.append(Particle.velocity + Particle.active_velocity)





def beads_distance(formerpos, desireddist):
    success = False
    
    while (success == False):
    
        a= np.random.uniform(-desireddist, desireddist)
        b = np.random.uniform(-desireddist+abs(a),desireddist-abs(a))
        c = np.sqrt(desireddist**2 - a**2 - b**2)
        lengthvec = np.linalg.norm(np.array([a,b,c]))
        newdist = np.array([a,b,c])
        newpoint = formerpos + newdist

        if impl_obstacles == True:
            currobstacles = []
            for coord in newpoint:
                multfact = (coord-boxa)/(obstaclenr)
                obsposs1 = boxa+int(multfact)*obstaclenr
                obsposs2 = boxa+(int(multfact)+1)*obstaclenr

                currobstacles.append([obsposs1, obsposs2])
                
                
                
            newobstaclelist = list(itertools.product(*currobstacles))
  
            allfar = True
            for obstacle in newobstaclelist:

                newcoord = np.array(obstacle)
               # connvector = newcoord - newpoint


                obstacledist = np.linalg.norm(obstacle-newpoint)
                obstacledist = obstacledist - obstacle_radius

                if (obstacledist < 0):
                    allfar = False
                    break
            if (allfar == True):
                success = True
                
        
        else:
            success = True

                    
    
    return newpoint




def beads_distance_simple(formerpos, desireddist):


    a= np.random.uniform(-desireddist, desireddist)
    b = np.random.uniform(-desireddist+abs(a),desireddist-abs(a))
    c = np.sqrt(desireddist**2 - a**2 - b**2)
    lengthvec = np.linalg.norm(np.array([a,b,c]))
    newdist = np.array([a,b,c])
    newpoint = formerpos + newdist
                

    return newpoint


def beads_distance_3d(formerpos, desireddist, all_particles):
    success = False
    
    while (success == False):
    
        a= np.random.uniform(-desireddist, desireddist)
        b = np.random.uniform(-desireddist+abs(a),desireddist-abs(a))
        c = np.sqrt(desireddist**2 - a**2 - b**2)
        lengthvec = np.linalg.norm(np.array([a,b,c]))
        newdist = np.array([a,b,c])
        newpoint = formerpos + newdist

        success = True
        for other_particle in all_particles:
            particledist = np.linalg.norm(other_particle.position-newpoint)
            if particledist < desireddist:
                success = False
                break

    
    return newpoint




import numpy as np

def beads_distance_2d(formerpos, desireddist, all_particles, max_attempts=1000000):
    success = False
    attempts = 0
    
    while not success and attempts < max_attempts:
        # Uniformly sample angle theta between 0 and 2pi
        theta = np.random.uniform(0, 2 * np.pi)
        
        # Calculate the new point at distance desireddist from formerpos
        newpoint = formerpos + np.array([desireddist * np.cos(theta), desireddist * np.sin(theta)])

        # Check for collisions with existing particles
        success = all(np.linalg.norm(other_particle.position - newpoint) >= desireddist
                      for other_particle in all_particles)
        
        attempts += 1

    if not success:
        raise ValueError("Couldn't find a valid position after many attempts.")

    return newpoint



def random_vector_with_length(length):
    # Generate random angles
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    
    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # Create the unit vector
    unit_vector = np.array([x, y, z])
    
    # Scale the unit vector to the desired length
    scaled_vector = unit_vector * length
    
    return scaled_vector




import random

def create_list_of_lists(num_entries, entry_lengths):
    """
    Creates a list of lists where each sublist contains sequential numbers.
    
    :param num_entries: Number of sublists.
    :param entry_lengths: Either an integer (fixed length for all sublists) or a list of integers (variable lengths).
    :return: A list of lists with sequential numbers.
    """
    list_of_lists = []
    current_number = 0

    # If a single integer is provided, create uniform-length sublists
    if isinstance(entry_lengths, int):
        entry_lengths = [entry_lengths] * num_entries

    for length in entry_lengths:
        sublist = list(range(current_number, current_number + length))
        list_of_lists.append(sublist)
        current_number += length  # Ensure continuity
    
    return list_of_lists

def select_random_numbers(list_of_lists):
    """
    Selects a random number from each sublist.
    
    :param list_of_lists: A list of lists containing sequential numbers.
    :return: A list of randomly selected numbers.
    """
    return [random.choice(sublist) for sublist in list_of_lists]





def init_particles_partners_simple_central_sab(temp = [0.1,0.1,0.2,0.2], entities=2, cpart_type="sab", 
                                               active_velocity0=1, move_groups_nr=5, furtherparticles=50, free_particles=None, 
                                               random_distr_of_move_groups=False, desireddist = 0.8, friction = 1, explicit_groups=None):
    import math 

    global particles
    boxa = -2
    boxe = 2

    particles = []

    #entity_temps=[0.1,0.1,0.2,0.2]

    if isinstance(temp, float) or isinstance(temp, int):
        entity_temps = [temp for x in range(entities)]
    else:
        entity_temps = temp


    num_entries = move_groups_nr
    #fixed_length = 4
    fixed_length = math.floor(furtherparticles / move_groups_nr)


    list_of_lists_fixed = create_list_of_lists(num_entries, fixed_length)
    print("Fixed-length list of lists:", list_of_lists_fixed)
    print("Randomly selected numbers:", select_random_numbers(list_of_lists_fixed))

    central_particles = select_random_numbers(list_of_lists_fixed)

    initvel = np.array([0 for _ in range(partdim)])
    initforce = np.array([0 for _ in range(partdim)])
    initfriction = np.array([friction for _ in range(partdim)])

    successful_initialization = False

    while not successful_initialization:
        try:
            particles = []
            #active_velocity_vec=np.array([1,1,1])
            init_active_velocity_vec = np.array([random.uniform(-1, 1) for _ in range(partdim)])
            for ip, particle_group in enumerate(list_of_lists_fixed):
                initpos = np.array([random.uniform(boxa, boxe) for _ in range(partdim)])
                nextpos = initpos
                for iip, particle_entry in enumerate(particle_group):    
                    if partdim == 3:
                        nextpos = beads_distance_3d(nextpos, desireddist, particles)
                    elif partdim == 2:
                        nextpos = beads_distance_2d(nextpos, desireddist, particles)
                    else:
                        raise ValueError("Wrong number of dimensions.")

                    curr_group = particle_group.copy()
                    curr_group.remove(particle_entry)

                    if central_particles[ip] == particle_entry:
                        curr_central_particle = True
                    else:
                        curr_central_particle = False

                    if not curr_central_particle:
                        particles.append(Particle(nextpos,initvel,1,initfriction,initforce,1, active_velocity_vec=init_active_velocity_vec, identity = particle_entry, interaction_partners=curr_group, interaction_group=ip, central_particle = curr_central_particle, central_particle_group= central_particles[ip], group=ip, entity_group=ip, move_group=ip, number=particle_entry))    
                    else:
                        if cpart_type == "sab":
                            particles.append(Particle(nextpos,initvel,1,initfriction,initforce,1, active_velocity_vec=init_active_velocity_vec,active_velocity0=active_velocity0 ,particle_type="sab", sab=True, identity = particle_entry, interaction_partners=curr_group, interaction_group=ip, central_particle = curr_central_particle, central_particle_group= central_particles[ip], group=ip, entity_group=ip, move_group=ip, number=particle_entry))    
                        elif cpart_type == "const":
                            particles.append(Particle(nextpos,initvel,1,initfriction,initforce,1, active_velocity_vec=init_active_velocity_vec,active_velocity0=active_velocity0 ,particle_type="const", sab=False, identity = particle_entry, interaction_partners=curr_group, interaction_group=ip, central_particle = curr_central_particle, central_particle_group= central_particles[ip], group=ip, entity_group=ip, move_group=ip, number=particle_entry))    
                        else:
                            particles.append(Particle(nextpos,initvel,1,initfriction,initforce,1, active_velocity_vec=init_active_velocity_vec,active_velocity0=active_velocity0 ,particle_type="simple", identity = particle_entry, interaction_partners=curr_group, interaction_group=ip, central_particle = curr_central_particle, central_particle_group= central_particles[ip], group=ip, entity_group=ip, move_group=ip, number=particle_entry))    
            successful_initialization = True
        except (ValueError, IndexError) as e:
            print(f"Initialization error: {e}. Retrying...")  



    return particles






def init_particles_partners_simple_central_allsab(temp = [0.1,0.1,0.2,0.2], entities=2, cpart_type="sab", 
                                               active_velocity0=1, move_groups_nr=5, furtherparticles=50, free_particles=None, 
                                               random_distr_of_move_groups=False, desireddist = 0.8, friction = 1, explicit_groups=None):
    import math 

    global particles
    boxa = -2
    boxe = 2

    particles = []

    #entity_temps=[0.1,0.1,0.2,0.2]




    num_entries = move_groups_nr
    #fixed_length = 4
    fixed_length = math.floor(furtherparticles / move_groups_nr)


    list_of_lists_fixed = create_list_of_lists(num_entries, fixed_length)
    print("Fixed-length list of lists:", list_of_lists_fixed)
    print("Randomly selected numbers:", select_random_numbers(list_of_lists_fixed))

    if isinstance(temp, float) or isinstance(temp, int):
        entity_temps = [temp for x in range(len(list_of_lists_fixed))]
    else:
        entity_temps = temp


    central_particles = select_random_numbers(list_of_lists_fixed)

    initvel = np.array([0 for _ in range(partdim)])
    initforce = np.array([0 for _ in range(partdim)])
    initfriction = np.array([friction for _ in range(partdim)])

    successful_initialization = False

    while not successful_initialization:
        try:
            particles = []
            #active_velocity_vec=np.array([1,1,1])
            init_active_velocity_vec = np.array([random.uniform(-1, 1) for _ in range(partdim)])
            for ip, particle_group in enumerate(list_of_lists_fixed):
                initpos = np.array([random.uniform(boxa, boxe) for _ in range(partdim)])
                nextpos = initpos
                for iip, particle_entry in enumerate(particle_group):    
                    if partdim == 3:
                        nextpos = beads_distance_3d(nextpos, desireddist, particles)
                    elif partdim == 2:
                        nextpos = beads_distance_2d(nextpos, desireddist, particles)
                    else:
                        raise ValueError("Wrong number of dimensions.")

                    curr_group = particle_group.copy()
                    curr_group.remove(particle_entry)

                    if central_particles[ip] == particle_entry:
                        curr_central_particle = True
                    else:
                        curr_central_particle = False

                    if not curr_central_particle:
                        particles.append(Particle(nextpos,initvel,1,initfriction,initforce,entity_temps[ip], active_velocity_vec=init_active_velocity_vec,active_velocity0=active_velocity0 ,particle_type="sab_follow", sab=True, identity = particle_entry, interaction_partners=curr_group, interaction_group=ip, central_particle = curr_central_particle, central_particle_group= central_particles[ip], group=ip, entity_group=ip, move_group=ip, number=particle_entry))    
                    else:
                        if cpart_type == "sab":
                            particles.append(Particle(nextpos,initvel,1,initfriction,initforce,entity_temps[ip], active_velocity_vec=np.array([random.uniform(-1, 1) for _ in range(partdim)]),active_velocity0=active_velocity0 ,particle_type="sab", sab=True, identity = particle_entry, interaction_partners=curr_group, interaction_group=ip, central_particle = curr_central_particle, central_particle_group= central_particles[ip], group=ip, entity_group=ip, move_group=ip, number=particle_entry))    
                        elif cpart_type == "const":
                            particles.append(Particle(nextpos,initvel,1,initfriction,initforce,entity_temps[ip], active_velocity_vec=init_active_velocity_vec,active_velocity0=active_velocity0 ,particle_type="const", sab=False, identity = particle_entry, interaction_partners=curr_group, interaction_group=ip, central_particle = curr_central_particle, central_particle_group= central_particles[ip], group=ip, entity_group=ip, move_group=ip, number=particle_entry))    
                        else:
                            particles.append(Particle(nextpos,initvel,1,initfriction,initforce,entity_temps[ip], active_velocity_vec=init_active_velocity_vec,active_velocity0=active_velocity0 ,particle_type="simple", identity = particle_entry, interaction_partners=curr_group, interaction_group=ip, central_particle = curr_central_particle, central_particle_group= central_particles[ip], group=ip, entity_group=ip, move_group=ip, number=particle_entry))    
            successful_initialization = True
        except (ValueError, IndexError) as e:
            print(f"Initialization error: {e}. Retrying...")  



    return particles



def init_particles_partners_simple_central_allsab_global_vel(temp = [0.1,0.1,0.2,0.2], entities=2, cpart_type="sab", 
                                               active_velocity0=1, move_groups_nr=5, furtherparticles=50, free_particles=None, 
                                               random_distr_of_move_groups=False, desireddist = 0.8, friction = 1, explicit_groups=None):
    import math 

    global particles
    boxa = -2
    boxe = 2

    particles = []

    #entity_temps=[0.1,0.1,0.2,0.2]




    num_entries = move_groups_nr
    #fixed_length = 4
    fixed_length = math.floor(furtherparticles / move_groups_nr)


    list_of_lists_fixed = create_list_of_lists(num_entries, fixed_length)
    print("Fixed-length list of lists:", list_of_lists_fixed)
    print("Randomly selected numbers:", select_random_numbers(list_of_lists_fixed))

    if isinstance(temp, float) or isinstance(temp, int):
        entity_temps = [temp for x in range(len(list_of_lists_fixed))]
    else:
        entity_temps = temp


    central_particles = select_random_numbers(list_of_lists_fixed)

    initvel = np.array([0 for _ in range(partdim)])
    initforce = np.array([0 for _ in range(partdim)])
    initfriction = np.array([friction for _ in range(partdim)])

    #active_velocity_vec=np.array([1,1,1])
    init_active_velocity_vec = np.array([random.uniform(-1, 1) for _ in range(partdim)])
    for ip, particle_group in enumerate(list_of_lists_fixed):
        initpos = np.array([random.uniform(boxa, boxe) for _ in range(partdim)])
        nextpos = initpos
        for iip, particle_entry in enumerate(particle_group):    
            if partdim == 3:
                nextpos = beads_distance_3d(nextpos, desireddist, particles)
            elif partdim == 2:
                nextpos = beads_distance_2d(nextpos, desireddist, particles)
            else:
                raise ValueError("Wrong number of dimensions.")

            curr_group = particle_group.copy()
            curr_group.remove(particle_entry)

            if central_particles[ip] == particle_entry:
                curr_central_particle = True
            else:
                curr_central_particle = False


            if cpart_type == "sab":
                particles.append(Particle(nextpos,initvel,1,initfriction,initforce,entity_temps[ip], active_velocity_vec=np.array([random.uniform(-1, 1) for _ in range(partdim)]),active_velocity0=active_velocity0 ,particle_type="sab", sab=True, identity = particle_entry, interaction_partners=curr_group, interaction_group=ip, central_particle = curr_central_particle, central_particle_group= central_particles[ip], group=ip, entity_group=ip, move_group=ip, number=particle_entry))    
            elif cpart_type == "const":
                particles.append(Particle(nextpos,initvel,1,initfriction,initforce,entity_temps[ip], active_velocity_vec=init_active_velocity_vec,active_velocity0=active_velocity0 ,particle_type="const", sab=False, identity = particle_entry, interaction_partners=curr_group, interaction_group=ip, central_particle = curr_central_particle, central_particle_group= central_particles[ip], group=ip, entity_group=ip, move_group=ip, number=particle_entry))    
            else:
                particles.append(Particle(nextpos,initvel,1,initfriction,initforce,entity_temps[ip], active_velocity_vec=init_active_velocity_vec,active_velocity0=active_velocity0 ,particle_type="simple", identity = particle_entry, interaction_partners=curr_group, interaction_group=ip, central_particle = curr_central_particle, central_particle_group= central_particles[ip], group=ip, entity_group=ip, move_group=ip, number=particle_entry))    



    return particles


group_velocity = None
thermostat_for_group = False




def particles_init_simulate_smallgroups(stim_steps=1000, move_groups_nr=5, furtherparticles=50, dt = 0.01, act_magnitude=1, box_variable = False, boxl_variable=50, replace_vel_variable=True, adopt_vel_var=True, adopt_act_val_var=False, partdim_variable=3, cpart_type="const", only_central_interactions_variable = True, const_vel_try_variable = False, interaction_partner_calculation = True,  group_thermostat=False, harmpot=1, thermostat=1, fenepot=0):
    global harmpot_on
    global fenepot_on
    global thermostat_on
    harmpot_on = harmpot
    fenepot_on = fenepot
    thermostat_on = thermostat
    global thermostat_for_group
    thermostat_for_group = group_thermostat

    global adopt_vel
    global adopt_act_vel

    adopt_vel = adopt_vel_var
    adopt_act_vel = adopt_act_val_var


    global steps
    global timestep 

    timestep = dt
    steps = stim_steps

    global const_vel_try
    const_vel_try = const_vel_try_variable

    global only_central_interactions
    only_central_interactions = only_central_interactions_variable

    global partdim
    partdim = partdim_variable

    global replace_vel
    replace_vel = replace_vel_variable

    global boxl
    boxl = boxl_variable



    global interaction_partner_calc
    interaction_partner_calc = interaction_partner_calculation

    global box
    box = box_variable

    particles = init_particles_partners_simple_central_sab(cpart_type=cpart_type, active_velocity0=act_magnitude, move_groups_nr=move_groups_nr, furtherparticles=furtherparticles)

    run_simulation(particles, partners = True)


    return particles




def particles_init_simulate_smallgroups_sab(stim_steps=1000, change_groups_after_steps_var=100, move_groups_nr=5, furtherparticles=50, T=1, dt = 0.01, act_magnitude=1, box_variable = False, boxl_variable=50, replace_vel_variable=True, adopt_vel_var=True, adopt_act_val_var=False, partdim_variable=3, cpart_type="const", only_central_interactions_variable = True, const_vel_try_variable = False, interaction_partner_calculation = True,  group_thermostat=False, harmpot=1, thermostat=1, fenepot=0):
    global harmpot_on
    global fenepot_on
    global thermostat_on
    harmpot_on = harmpot
    fenepot_on = fenepot
    thermostat_on = thermostat
    global thermostat_for_group
    thermostat_for_group = group_thermostat

    global adopt_vel
    global adopt_act_vel

    adopt_vel = adopt_vel_var
    adopt_act_vel = adopt_act_val_var


    global steps
    global timestep 

    timestep = dt
    steps = stim_steps

    global const_vel_try
    const_vel_try = const_vel_try_variable

    global only_central_interactions
    only_central_interactions = only_central_interactions_variable

    global partdim
    partdim = partdim_variable

    global replace_vel
    replace_vel = replace_vel_variable

    global boxl
    boxl = boxl_variable

    global change_groups_after_steps
    change_groups_after_steps = change_groups_after_steps_var



    global interaction_partner_calc
    interaction_partner_calc = interaction_partner_calculation

    global box
    box = box_variable

    particles = init_particles_partners_simple_central_allsab(cpart_type="sab", temp=T, active_velocity0=act_magnitude, move_groups_nr=move_groups_nr, furtherparticles=furtherparticles)

    run_simulation_sab_follow(particles)

#def init_particles_partners_simple_central_sab(temp = [0.1,0.1,0.2,0.2], entities=2, cpart_type="sab", move_groups_nr=5, furtherparticles=50, free_particles=None, random_distr_of_move_groups=False, desireddist = 0.8, friction = 1, explicit_groups=None):

    return particles



def particles_init_simulate_smallgroups_sab_velocity_objects(stim_steps=1000, Dvar=0.1, update_global_activity_vector_var = True, change_groups_after_steps_var=100, invert_velocity_after_steps_var=50, move_groups_nr=5, furtherparticles=50, T=1, dt = 0.01, act_magnitude=1, box_variable = False, boxl_variable=50, replace_vel_variable=True, adopt_vel_var=True, adopt_act_val_var=False, partdim_variable=3, cpart_type="const", only_central_interactions_variable = True, const_vel_try_variable = False, interaction_partner_calculation = True,  group_thermostat=False, harmpot=1, thermostat=1, fenepot=0):
    global D
    D = Dvar
    global update_global_activity_vector
    update_global_activity_vector = update_global_activity_vector_var
    
    global harmpot_on
    global fenepot_on
    global thermostat_on
    harmpot_on = harmpot
    fenepot_on = fenepot
    thermostat_on = thermostat
    global thermostat_for_group
    thermostat_for_group = group_thermostat

    global adopt_vel
    global adopt_act_vel

    adopt_vel = adopt_vel_var
    adopt_act_vel = adopt_act_val_var


    global steps
    global timestep 

    timestep = dt
    steps = stim_steps

    global const_vel_try
    const_vel_try = const_vel_try_variable

    global only_central_interactions
    only_central_interactions = only_central_interactions_variable

    global partdim
    partdim = partdim_variable

    global replace_vel
    replace_vel = replace_vel_variable

    global boxl
    boxl = boxl_variable

    global change_groups_after_steps
    change_groups_after_steps = change_groups_after_steps_var

    global invert_velocity_after_steps
    invert_velocity_after_steps = invert_velocity_after_steps_var

    global interaction_partner_calc
    interaction_partner_calc = interaction_partner_calculation

    global box
    box = box_variable

    global act_velocity_vectors
    global act_velocity_vectors_magnitudes
    act_velocity_vectors = []
    act_velocity_vectors_magnitudes = []
    for group_vel in range(move_groups_nr):
        active_velocity_vec = random_3d_vector_with_magnitude(magnitude=1)
        act_velocity_vectors.append(active_velocity_vec)

        #in case of uniform magnitudes
        act_velocity_vectors_magnitudes.append(act_magnitude)

 

    particles = init_particles_partners_simple_central_allsab_global_vel(cpart_type="sab", temp=T, active_velocity0=act_magnitude, move_groups_nr=move_groups_nr, furtherparticles=furtherparticles)

    run_simulation_sab_follow_global_vels(particles)

    return particles

