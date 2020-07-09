# === IMPORTS ===
import numpy as np
verbosity = 0 # {0: no prints, 1: initializations, 2: essentials, 3: everything}

# === Object ===
class COVID_Model:
    """
    Attributes
    ----------
    n_persons : int
        # people
    base_infectiousness : float
        unit dose/unit time of close proximity
    close_contact_prob : float
        fraction of time together that counts as infectiously-close proximity, given that two people are in the same space
    dt : float
        time step (unit: days)
    time : float

    dose_matrix : 2D array
    contact_history : 2D array
    infection_history : 2D array

    members_in_group : list
    group_id : 1D array
    time_left_in_EII : 1D array
    t0_infection : 1D array
        treat negative as not-infected

    indv_factor : 1D array
        catch-all to account for, e.g., asymptomatic reduced infectiousness, mask wearing, etc.
    symptomatic_if_infected: 1D array
    indv_infectiousness : 1D array

    symptomatic : 1D array
    quarantined : 1D array
    indv_infectiousness : 1D array
       

    Methods
    -------
    assign_group_random()
    assign_work_random()     
    disease_history_simple()
    quarantine_simple()
    background_update_simple()


    Notes
    -----
    Currently assigns people to groups randomly, treats everyone's infectiousness multiplier as the same

    Protocols that can/should be customized:
    1) assign people to groups
    2) assign/schedule people to E-II
    3) quarantine protocol
    4) disease history
    5) background infection rate

    Wishlist:
    0) Possibly turn the "protocol" functions into acting on inputs and returning outputs, instead of explicitly acting on object attributes
    1) Draw individual infectiousness factors from a distribution
    2) Better work assignment protocol, e.g. time-varying throughout the day, distribution of lab times, etc.
    3) Better treatment of base infectiousness if people aren't in same lab? (i.e. infectiousness from common spaces)
    4) Quarantining people in contact with infected people, but not infected themselves
    5) Coupling in SB dynamics (currently SB is a constant background rate for when people are out of lab)
    6) Lab shutdowns
    7) Individual "resistance factor" (i.e. PPE effects on preventing inhaling virus, not just reducing virus emission)  
    8) Measuring R0 for an individual... (perhaps have to seed one person, and repeat the experiment multiple times
    """

    def __init__(self, n_persons, beta0, contact_prob, dt=1.0, init_protocol=0, asymptomatic_rate = 0.25):
        # === store variables ===
        self.n_persons = n_persons

        self.base_infectiousness = beta0
        self.close_contact_prob = contact_prob

        self.time = 0
        self.dt = dt
        self.asymptomatic_rate = asymptomatic_rate

        # === hard-coded variables that we haven't implemented I/O for yet ===
        self.asymptomatic_factor = 0.5
        self.EII_background_rate = 0.01 #infectious dose/24hr/infectious person
        self.SB_background_rate  = 0.005 * 1 / 3 #infectious dose/day, roughly proportional to (%SB_population_active_case * R / days of infection)
        self.EII_background = 0. #will be updated each step
        self.SB_background = 0.

        # === initialize parameters ===
        self.group_id = np.zeros( self.n_persons )
        self.group_contact_matrix = np.ones( (self.n_persons, self.n_persons) )
        self.time_left_in_EII = np.zeros( self.n_persons )
        self.quarantined = np.zeros( self.n_persons, dtype=bool )

        self.t0_infection = -1.0 * np.ones( self.n_persons )
        self.community_infection = np.zeros( self.n_persons, dtype=bool )
        self.symptomatic_if_infected = np.random.choice([True, False], size=self.n_persons, p=[1-asymptomatic_rate, asymptomatic_rate])
        self.symptomatic = np.zeros( self.n_persons, dtype=bool )
        self.indv_factor = np.ones( self.n_persons )
        self.indv_factor[ self.symptomatic_if_infected ] = self.asymptomatic_factor
        self.indv_infectiousness = np.zeros(self.n_persons)

        if init_protocol == 0: #simple random initializaton protocol
            if verbosity > 0: print("=== Simple random initialization protocol ===")
            self.init_simple()
        else:
            if verbosity > 0: print("=== System not initialized, assuming everyone in contact with everyone all the time ===")


    def init_simple(self):
        self.assign_group_random()
        #possibly draw individual factors from a distribution?

    def assign_group_random(self, avg_lab_size = 10):
        """Assigns people to groups with prescribed average lab size, randomly
        """
        if verbosity > 0: print('...assigning people to random groups of average size {}'.format(avg_lab_size))
        self.group_contact_matrix = np.zeros( (self.n_persons, self.n_persons) )
        n_groups = int(np.floor( self.n_persons/avg_lab_size ))
        self.group_id = np.random.randint(0, n_groups, self.n_persons)
        self.members_in_group = []

        for ii in range(n_groups):
            members = np.argwhere( self.group_id == ii ).flatten()
            self.members_in_group.append( members )
            if verbosity > 0: print('group {} members: {}'.format(ii,members))
            self.group_contact_matrix[ np.ix_(members,members) ] = 1


    def assign_work_random(self, avg_lab_time = 1.0, fraction_working = 0.1, lab_constraint = 0):
        """Assigns people to work, randomly. Assume common lab time for everyone.
        Parameters
        ----------
        avg_lab_time : float
        fraction_working : fraction of people allowed to be in E-II at any one time
        lab_constraint : int
            maximum # people from same group allowed to be in contact. 0 = no constraint.

        Notes
        -----
        Modifies self.time_left_in_EII to reflect the amount of working time left for the person
        I/O:
            modifies self.time_left_in_EII, essentially who is in EII
        """
        if verbosity > 2: print('\n--- Assigning people to EII ---')
        max_num_working = int( np.floor(fraction_working * self.n_persons) )
        mask_currently_working = (self.time_left_in_EII > 0)
        num_currently_working = mask_currently_working.sum()

        while num_currently_working < max_num_working: #assign more people to work
            # choose random person
            healthy_nonworking = (~mask_currently_working) * (~self.quarantined)
            new_worker = np.random.choice( np.argwhere( healthy_nonworking ).flatten() )

            # check lab constraints aren't violated
            num_currently_in_lab = (self.group_id[mask_currently_working] == self.group_id[new_worker]).sum()
            if lab_constraint > 0:
                if num_currently_in_lab > lab_constraint:
                    #chose someone whose lab is full, pass
                    continue

            # assign worker to EII
            self.time_left_in_EII[new_worker] = avg_lab_time
            mask_currently_working[new_worker] = True
            num_currently_working += 1
        
        if verbosity > 2: print("...People's time left to work in EII: {}".format( self.time_left_in_EII ))
        if verbosity > 2: print('...Num working: {}'.format(  mask_currently_working.sum()))
        if verbosity > 2: print('...Num at home: {}'.format((~mask_currently_working).sum()))


    def disease_history_simple(self):
        """Simple model of diease history, infectiousness, symptoms, etc.
        Returns
        -------
        infectiousness : 1D array

        Notes
        -----
        Assuming simple linear ramps, based on He et al. 2020, https://www.nature.com/articles/s41591-020-0869-5.pdf
        Infectiousness: ramp to peak infectiousness in 2 days, 7 day decline to minimal shedding
        Symptoms: days 3~14

        I/O: 
            modifies self.indv_infectiousness (scaled 0~1)
            modifies self.symptomatic
        """
        if verbosity > 2: print('\n--- Assigning disease history/progression, infectiousness, symptoms ---')
        shedding_peak = 2 #units: days
        shedding_end  = 9
        symptom_onset = 3
        symptom_end   = 14

        self.indv_infectiousness = np.zeros( self.n_persons )
        time_spent_infected = self.time - self.t0_infection

        actively_shedding = (self.t0_infection > 0) * (time_spent_infected < shedding_end)
        self.indv_infectiousness[ time_spent_infected <= shedding_peak ] = time_spent_infected[ time_spent_infected <= shedding_peak ] / shedding_peak
        self.indv_infectiousness[ time_spent_infected > shedding_peak ] = 1.0 - time_spent_infected[ time_spent_infected > shedding_peak ] / (shedding_end-shedding_peak)
        self.indv_infectiousness[~actively_shedding] = 0

        self.symptomatic[ (self.t0_infection > 0) * (time_spent_infected >= symptom_onset) * (time_spent_infected < symptom_end) ] = True
        self.symptomatic *= self.symptomatic_if_infected

        if verbosity > 2: print('...indv_infectiousness: {}'.format(self.indv_infectiousness))
        if verbosity > 2: print('...symptomatics: {}'.format(self.symptomatic))

    def quarantine_simple(self):
        """Simple quarantine protocol: only quarantine infected, symptomatic people
        Notes
        -----
        I/O:
            modifies self.quarantined
        """
        if verbosity > 2: print('\n--- Quarantining symptomatics ---')
        self.quarantined[ self.symptomatic ] = True


    def background_update_simple(self):
        """ Update the background infection rate
        Returns
        -------
        EII_background : float
        SB_background : float

        Notes
        -----
        Right now treats SB as constant. Adds up infectious people in EII and scales by some factor.
        """
        if verbosity > 2: print('\n--- Updating background exposure ---')
        mask_currently_working = (self.time_left_in_EII > 0)
        EII_background = self.EII_background_rate \
                  * ( self.indv_infectiousness[ mask_currently_working ]
                  * self.indv_factor[ mask_currently_working ]
                  * (~self.quarantined[ mask_currently_working ]) ).sum()
        if verbosity > 2: print('...EII background: {}\n...SB background: {}'.format(EII_background, self.SB_background_rate))
        return EII_background, self.SB_background_rate


    def propagate(self):
        """Exponential dose-response propagation dynamics. for EII
        Notes
        -----
        This section should be fairly constant/not change much... the main thing to modify is the background rates
        """
        if verbosity > 2: print('\n--- Propagating infection dynamics ---')
        mask_currently_working = (self.time_left_in_EII > 0)
        in_EII_matrix = np.outer( mask_currently_working, mask_currently_working )

        # === Calculate infection dose each person is exposed to ===    
        force_of_infection = self.dt * self.base_infectiousness * self.close_contact_prob \
                * (self.group_contact_matrix * in_EII_matrix) @ (self.indv_infectiousness * self.indv_factor * (~self.quarantined))

        # ... and account for background infection rate
        force_of_infection[ mask_currently_working] += self.dt * self.EII_background #background infection rate in hallways, common areas
        force_of_infection[~mask_currently_working] = self.dt * self.SB_background #background infection rate in greater SB community

        if verbosity > 2: print('...exposed individuals: {}'.format(np.argwhere(force_of_infection != 0.0).T))

        # === Propagate Infection ===
        p_infect = 1 - np.exp(-force_of_infection)
        rand_nums = np.random.rand(self.n_persons)

        proposed_infection = np.argwhere( rand_nums < p_infect )
        if verbosity > 1: print('...Proposed infections: {}'.format(proposed_infection.T))

        #filter to make sure is new infection
        new_infection = proposed_infection[ self.t0_infection[proposed_infection] < 0 ]
        self.community_infection[ new_infection ] = np.where( ~mask_currently_working[new_infection], True, False ) #True if new_infection is outside work, False otherwise
        if verbosity > 1: print('...New infection indices: {}'.format(new_infection))
        if verbosity > 1: print('...Community infection indices: {}'.format( [index for index in new_infection if self.community_infection[index]] ) )

        # update
        self.t0_infection[new_infection] = self.time
        self.time_left_in_EII[ self.time_left_in_EII > 0 ] -= self.dt
        self.time_left_in_EII[ self.time_left_in_EII <= 0.0 ] = 0.0
        self.time += self.dt

    def step(self, disease_history=None, quarantine=None, assign_work=None, background_update=None):
        """Perform one step of the disease propagation dynamics. Envision allowing swapping custom protocols
        """
        if verbosity > 1: print('\n===== TIME {} ====='.format(self.time))
        #  === Define protocol (just to make more apparent what functions to change) ===
        if disease_history is None:
            disease_history_protocol = self.disease_history_simple
        if quarantine is None:
            quarantine_protocol = self.quarantine_simple
        if assign_work is None:
            assign_work_protocol = self.assign_work_random
        if background_update is None:
            background_update_protocol = self.background_update_simple

        # === Process disease progression ===
        disease_history_protocol()
        quarantine_protocol()

        # === Assign people to work ===
        assign_work_protocol()

        # === Run update ===
        self.EII_background, self.SB_background = background_update_protocol()
        self.propagate()

        # === Record Keeping ===
        infected = np.nonzero( self.t0_infection >= 0 )
        infection_history = np.vstack( [infected, self.community_infection[infected], self.t0_infection[infected]] )
        sorted_history = np.argsort(infection_history[2,:])
        infection_history = infection_history[:,sorted_history]
        self.infection_history = infection_history

        # === Track R for each invidual ===
        # not sure how to allocate yet if there are multiple infectious people about...

        # === Misc ===
        return self.time, self.infection_history

