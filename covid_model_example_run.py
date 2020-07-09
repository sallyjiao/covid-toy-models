import numpy as np
import covid_model

fraction_working = 0.3
asymptomatic_rate = 0.25

# Initialize object and run one step with full verbosity for demonstration
covid_model.verbosity = 3
obj = covid_model.COVID_Model(500, 0.25, 0.25, dt=1.0, init_protocol = 0, asymptomatic_rate=asymptomatic_rate)
obj.step( assign_work = obj.assign_work_random(fraction_working=fraction_working) )

# Run model for 201 more days:
covid_model.verbosity = 0
for iters in range(200+1):
    time,history = obj.step( assign_work = obj.assign_work_random(fraction_working=fraction_working) )
    if np.mod(iters,100) == 0:
        print('time: {} \n...history: {}'.format(time,history))

# Basic analysis:
print('\n')
print('Total infections: {}'.format((obj.t0_infection > 0).sum()) )
print('Number of community infections: {}'.format(obj.community_infection.sum()))

