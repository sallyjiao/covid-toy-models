import numpy as np
import covid_model
import matplotlib.pyplot as plt

fraction_working = 0.3
asymptomatic_rate = 0.25
n_people = 500
covid_model.verbosity = 2
base_infectiousness = 0.25
contact_prob = 0.25

n_model_runs = 1
n_days_max = 20

for i_run in range(n_model_runs):
    # initialize model
    obj = covid_model.COVID_Model(n_people, 
                                  base_infectiousness,
                                  contact_prob,
                                  dt=1.0, 
                                  init_protocol = 0, 
                                  asymptomatic_rate=asymptomatic_rate,
                                  reporter_names=['incidence'])

    # set background factors to 0
    obj.EII_background_rate = 0.0
    obj.SB_background_rate = 0.0

    # only person 0 is infectious if infected
    obj.indv_factor = np.zeros( obj.n_persons, dtype=bool )
    obj.indv_factor[0] = 1

    # person 0 is asymptomatic (so does not quarantine--this is in lieu of changing the quarantine policy)
    obj.symptomatic_if_infected[0] = False

    # 1 day
    obj.step( assign_work = obj.assign_work_random, 
              fraction_working=fraction_working )
    obj.step( assign_work = obj.assign_work_random, 
              fraction_working=fraction_working )

    # infect person 0
    obj.t0_infection[0] = 1.

    # run rest of the days
    for i_day in range(1, n_days_max):
        obj.step( assign_work = obj.assign_work_random, 
                  fraction_working=fraction_working )
        print('person 0 infectiousness', obj.indv_infectiousness[0])
    n_cases = np.sum(np.array(obj.reporters['incidence']))
    print('n_cases', n_cases)
    
