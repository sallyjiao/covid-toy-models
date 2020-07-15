import numpy as np
import covid_model
import time

fraction_working = 1.0
asymptomatic_rate = 0.25

SIR_beta = 0.
SIR_gamma = 0.

n_EII = 300


#not used
lab_shutdown_max = 3

# First, calibrate for workday unit
#   and beta0*contact_rate = infectious dose/wkday
#   if beta0 has units of, say, dose/hr, then contact_rate is hr_close_contact/wkday
# Initial calibration of the no-NPI model:
# R0-EII = 1.25
# R0-home = 1.25
# initial guess for home: 
#   tau_infect = 4 days
#   n_family   = 3
#   beta0*contact_prob*1workday = R0-home/tau_infectious/n_family = 5/48 infections/person/wkday
#   i.e. if contact_rate is 3 hr close contact at home/wkday, then
#   beta0_home ~ 5/48/contact_rate ~ 0.035 dose/hr close contact

# initial guess for work:
#   tau_infect = 4 days
#   n_group    = 10
#   over one work day: ~ 5/160 infections/person/wkday
#   take dt = 1 workday
#   guess close_contact_rate is 1hr/workday
#   beta0_work = 5/160/hour of close contact = 0.03125 dose/hr, close to guess for family, so consistent!

# community = 0.5
beta0 = 7/160. # infection rate per hr of close contact
beta0 = 0.05
contact_rate = 1.0 #hr close contact/wkday
dt = 1.0       # workday
asymptomatic_rate = 0.25

# further assume hallway rate is
#hallway_rate = (5/60/10)**2. #fraction of time in hallway * fraction of time in hallway
#EII_background_rate = beta0 * hallway_rate
EII_background_rate = beta0 * 0.001


n_trials = 1000
infection_counts = np.zeros(n_trials)
start = time.time()
for i_trial in range(n_trials):
    print('\n--- Trial {}, elapsed time {}s ---'.format(i_trial, time.time()-start))
    # === initialize ===
    covid_model.verbosity = 0
    obj = covid_model.COVID_Model(n_EII, beta0 = beta0, contact_prob=contact_rate, dt=1.0, init_protocol = 0, asymptomatic_rate=asymptomatic_rate)
    obj.community_s = 1.0
    obj.community_i = 0.0
    obj.community_r = 0.0

    obj.EII_background_rate = EII_background_rate
    obj.SB_background_rate = 0.0 #not used

    # seed one infection, and modifications to prevent spread past 1st generation
    obj.indv_factor[:] = 0.
    obj.t0_infection[0] = 1.
    obj.indv_factor[0] = 1.
    obj.symptomatic_if_infected[0] = True


    # 1 step
    obj.step( assign_work = obj.assign_work_random, quarantine=obj.quarantine_none, fraction_working=fraction_working, lab_shutdown_max=lab_shutdown_max, background_update=obj.background_update_communitySIR, beta=SIR_beta,gamma=SIR_gamma)

    # Run model for some more days:
    covid_model.verbosity = 0
    for iters in range(20+1):
        #time,history = obj.step( assign_work = obj.assign_work_random, quarantine=obj.quarantine_none, fraction_working=fraction_working, lab_shutdown_max=lab_shutdown_max, background_update=obj.background_update_communitySIR, beta=SIR_beta, gamma=SIR_gamma)
        current_time,history = obj.step( assign_work = obj.assign_work_random, quarantine=obj.quarantine_none, fraction_working=fraction_working, lab_shutdown_max=lab_shutdown_max, background_update=obj.background_update_communitySIR, beta=SIR_beta,gamma=SIR_gamma)
        #if np.mod(iters,100) == 0:
        #    print('time: {} \n...history: {}'.format(current_time,history))


    # Basic Analysis
    print('Total infections: {}'.format((obj.t0_infection > 0).sum()) )
    #print('Number of community infections: {}'.format(obj.community_infection.sum()))
    #print('Community SIR history:\n{}'.format(np.array(obj.community_history)))
    
    infection_counts[i_trial] = (obj.t0_infection > 0).sum()

print('=== Trial summary: ===')
print('avg # infections: {}'.format(infection_counts.mean()-1)) #don't include the seed
print('std deviation: {}'.format( (infection_counts-1).std()))

print('Histogram: {}'.format(np.histogram(infection_counts-1,bins=[0,1,2,3,4,5,6,7,8,9,10])))



