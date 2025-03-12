import itertools
import numpy as np


partdim = 3



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



#constants for friction term 
frica = 0.5
fricb = 5

feneK = hconst

#for harm potential
group_movement = False
polymer_movement = True


class Particle():
    newid = itertools.count().__next__
    def __init__(self, position, velocity, mass, friction , force , temperature, group=0, entity_group=0, move_group=0, full_group=0, number=0):
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.friction = friction
        self.force = force
        self.temperature = temperature
        
        self.positions = []
        self.velocities = []
        self.forces = []        
        self.kinetic1 = []
        self.potentiale = []
        self.frictions = []
        
        
        
        # additional friction for specific particles when using dynamic friction (can be assigned for each particle)
        self.frictionaadd = 2
        self.frictionbadd = 2
        self.frictionamult = 1
        self.frictionbmult= 1
        
        #friction at initialization
        self.startfriction = friction
        
        self.id = Particle.newid()
        
        #variables for 2nd order integrator
        self.c=0
        self.theta=0
        self.xi=0
        
        #counting of steps performed
        self.counter = 0

        self.group = group
        self.entity_group = entity_group
        self.move_group = move_group
        self.full_group = full_group

        self.number = number


    
#standard velocity verlet scheme: new positions
def new_pos(Particle):

       
    Particle.positions.append(Particle.position) 
    Particle.velocities.append(Particle.velocity) 

    Particle.position = Particle.position + Particle.velocity*timestep + (Particle.force )*(timestep**2)*0.5 / Particle.mass


#standard velocity verlet scheme: new velocities
def new_vel(Particle):

    actforces = new_forces_thermostat(Particle)  

    Particle.velocity = Particle.velocity + timestep * (actforces + Particle.force) / (2*Particle.mass)
    
    Particle.force = actforces



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
def new_forces_thermostat(Particle, min_group=1):
    force = 0
    force = force - Particle.friction * Particle.velocity * Particle.mass
    if (harmpot_on == 1):
        hp = harm_potential(Particle)
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







#quartic potential
def quartic_potential(Particle):
    pot = 0
    for ParticleP in particles:
        if (id(Particle) == id(ParticleP)):
            
            continue


        else: 
            dst = ParticleP.position-Particle.position
            r = np.linalg.norm(dst)
            vk = dst / np.linalg.norm(dst)
            
            pott = (hconst * 4 * r**3 ) 
            
            pot = pot - pott * vk 
    return pot





def run_simulation(particles, move_groups=None, group_additions = True, min_group=1):
    
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
            new_vel(Particle)

        if group_additions:
            compute_group_vel_additions(particles, move_groups, abs_vel=group_velocity)
            for Particle in particles:  
                curr_group = Particle.move_group

                Particle.velocity = Particle.velocity + additional_vels[curr_group-min_group] 






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


def beads_distance_full(formerpos, desireddist, all_particles):
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




def init_particles(temp = [0.1,0.1,0.2,0.2], entities=2, move_groups_nr=3, furtherparticles=20, free_particles=None, random_distr_of_move_groups=False, desireddist = 0.8, friction = 1, explicit_groups=None):
    global particle5
    global particle6
    global partdim
    global numpart
    global particles
    #initialization of particles:
    #arguments: 
    # position
    # velocity
    # mass
    # friction
    # intial force acting on particle (set to 0)
    # temperature (which is enforced by the thermostat, if switched on)

    # set friction coefficient for all particles    
    #setfriction = 1


    #3d
    #add all particles to a set
    particles = list()

    boxa = -2
    boxe = 2

    groups_nr = 2

    #entity_temps=[0.1,0.1,0.2,0.2]

    if isinstance(temp, float) or isinstance(temp, int):
        entity_temps = [temp for x in range(entities)]
    else:
        entity_temps = temp


        
    #initpos = np.array([1,1,-1])
    #nextpos = beads_distance(initpos, desireddist)

    
    div = np.linspace(0, furtherparticles, num=move_groups_nr+1)

    full_particle_counter = 0
    entity_particle_counter = 0
    start_move_group = 0

    pnr = 0
    for entity in range(entities):
        initpos = np.array([1+entity,1,-1])
        nextpos = beads_distance_full(initpos, desireddist, particles)

        for ip in range(furtherparticles):

            if not explicit_groups:
                group = ip%groups_nr
            else:
                group = explicit_groups[ip]

            #move_group = ip%move_groups_nr

            if not explicit_groups:
                if not random_distr_of_move_groups:
                    move_group = np.digitize(ip, div)
                else:
                    move_group = np.random.randint(0,move_groups_nr)

                move_group = move_group + start_move_group
            else:
                move_group = explicit_groups[ip]


            init_vel = np.array([1.5,1.5,1.5])


            particles.append(Particle(nextpos,init_vel,1,np.array([friction,friction,friction]),np.array([0,0,0]),entity_temps[entity], group=group, entity_group=entity, move_group=move_group, number=pnr))    
            pnr = pnr + 1
            nextpos = beads_distance_full(nextpos, desireddist, particles)

            full_particle_counter = full_particle_counter + 1

        entity_particle_counter = entity_particle_counter + 1
        start_move_group = move_group

    partdim = len(particles[0].position)
    

    numpart = len(particles)
    
    return particles



group_velocity = None
thermostat_for_group = False


def simulate_for_clustering(stim_steps=1000, dt = 0.01, group_thermostat=False, particles_per_group=20, move_groups_nr=3, group_vel = None, entities=2, temp=5, harmpot=1, thermostat=1, fenepot=0, desireddist=0.8, pot_const=1, group_m = False, polymer_m=True, group_additions=True, friction = 1, explicit_groups = None):
    global harmpot_on
    global fenepot_on
    global thermostat_on
    harmpot_on = harmpot
    fenepot_on = fenepot
    thermostat_on = thermostat

    global thermostat_for_group
    thermostat_for_group = group_thermostat



    global hconst
    global feneK
    hconst = pot_const
    feneK = pot_const

    #only for harm potential
    global group_movement
    global polymer_movement

    group_movement = group_m
    polymer_movement = polymer_m

    global steps
    global timestep 
    timestep = dt
    steps = stim_steps

    global group_velocity
    group_velocity = group_vel


    particles = init_particles(temp, entities=entities, move_groups_nr=move_groups_nr, desireddist=desireddist, friction = friction, furtherparticles=particles_per_group, explicit_groups = explicit_groups)
        


    move_groups = move_groups_nr * entities

    run_simulation(particles, move_groups=move_groups, group_additions=group_additions)

    return particles




