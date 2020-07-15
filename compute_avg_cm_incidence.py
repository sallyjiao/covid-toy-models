import numpy as np
import covid_model
import matplotlib.pyplot as plt

fraction_working = 0.3
asymptomatic_rate = 0.25
n_people = 500
covid_model.verbosity = 0

n_model_runs = 50
n_days_max = 100
block_size = 10 # in units of dt
cm_inc_growth_accum = np.zeros(n_days_max-1)
inc_accum = np.zeros(n_days_max)

lab_shutdown_max = 0

for i_run in range(n_model_runs):
    # initialize model
    obj = covid_model.COVID_Model(n_people, 
                                  0.25, 0.25, 
                                  dt=1.0, 
                                  init_protocol = 0, 
                                  asymptomatic_rate=asymptomatic_rate,
                                  reporter_names=['incidence'])

    for i_day in range(n_days_max):
        obj.step( assign_work = obj.assign_work_random, 
                  quarantine=obj.quarantine_lab_shutdown,
                  fraction_working=fraction_working,
                  lab_shutdown_max=lab_shutdown_max )

    # compute growth rate of cumulative incidence
    # cumulative incidence is the (# new cases) / (# at risk)
    # here, due to the significant asymptomatic population, we compute the apparent cumulative incidence
    # i.e. the number of new symptomatics / (population - number of people who have previously shown symptoms)
    incidence = np.array(obj.reporters['incidence'])
    inc_accum += incidence
    apparent_infected_pop = np.insert(np.cumsum(incidence), 0, [0])[:-1]
    cm_incidence = incidence / (n_people - apparent_infected_pop)
    cm_incidence_growth = (cm_incidence[1:] - cm_incidence[:-1]) / (cm_incidence[:-1] + 0.000001)
    cm_inc_growth_accum += cm_incidence_growth

    # computing a "coarse" growth rate of cumulative incidence by summing up incidence in blocks of multiple dt
    incidence_coarse = np.array([np.sum(incidence[i_block*block_size:(i_block+1)*block_size]) for i_block in range(int(n_days_max/block_size))])
    apparent_infected_pop = np.insert(np.cumsum(incidence_coarse), 0, [0])[:-1]
    cm_incidence_coarse = incidence_coarse / (n_people - apparent_infected_pop)
    cm_incidence_coarse_growth = (cm_incidence_coarse[1:] - cm_incidence_coarse[:-1]) / (cm_incidence_coarse[:-1] + 0.000001) / float(block_size)

cm_inc_growth_mean = cm_inc_growth_accum / float(n_model_runs)
plt.plot(cm_inc_growth_mean)

inc_accum /= float(n_model_runs)
apparent_infected_pop = np.insert(np.cumsum(inc_accum), 0, [0])[:-1]
cm_inc_accum = inc_accum / (n_people - apparent_infected_pop)
cm_inc_accum_growth = (cm_inc_accum[1:] - cm_inc_accum[:-1]) / (cm_inc_accum[:-1] + 0.000001)
plt.plot(cm_inc_accum_growth)
np.savetxt('inc_accum_labshutdown' + str(lab_shutdown_max) + '.txt', inc_accum)
np.savetxt('cm_inc_accum_growth_labshutdown' + str(lab_shutdown_max) + '.txt', cm_inc_accum_growth)

plt.ylim([0, 1])
plt.show()
