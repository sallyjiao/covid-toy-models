import numpy as np
import covid_model

fraction_working = 0.3
asymptomatic_rate = 0.25
lab_shutdown_max = 3

SIR_beta = 0.275 #choose such that beta/gamma = R0
SIR_gamma = 0.25 #average time before someone stops infecting


# Initialize object and run one step with full verbosity for demonstration
covid_model.verbosity = 2
obj = covid_model.COVID_Model(500, 0.25, 0.25, dt=1.0, init_protocol = 0, asymptomatic_rate=asymptomatic_rate)
obj.community_s = 0.99
obj.community_i = 0.01
obj.community_r = 0.00

obj.step( assign_work = obj.assign_work_random, quarantine=obj.quarantine_lab_shutdown, fraction_working=fraction_working, lab_shutdown_max=lab_shutdown_max, background_update=obj.background_update_communitySIR, beta=SIR_beta,gamma=SIR_gamma)

# Run model for some more days:
covid_model.verbosity = 2
for iters in range(100+1):
    time,history = obj.step( assign_work = obj.assign_work_random, quarantine=obj.quarantine_lab_shutdown, fraction_working=fraction_working, lab_shutdown_max=lab_shutdown_max, background_update=obj.background_update_communitySIR, beta=SIR_beta, gamma=SIR_gamma)
    if np.mod(iters,100) == 0:
        print('time: {} \n...history: {}'.format(time,history))

# Basic analysis:
print('\n')
print('Total infections: {}'.format((obj.t0_infection > 0).sum()) )
print('Number of community infections: {}'.format(obj.community_infection.sum()))

print('Community SIR history:\n{}'.format(np.array(obj.community_history)))

