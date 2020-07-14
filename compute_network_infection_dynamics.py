import numpy as np
import matplotlib.pyplot as plt

# returns tau matrix 
# tau[i, j]: average amount of time people i and j spend in "close" proximity in one timestep
# tau should be a symmetric matrix
# tau can be dependent on time and the specific people
# for now, is just a constant value-- Kevin will build this out
def tau(i_step, n_people):
    return 0.01 * np.ones((n_people, n_people))

# propagates infection by 1 timestep
# takes in a binary array of length n_people, 1=sick, 0=not sick
def step(is_infected, p_spread):
    n_people = len(is_infected)
    is_infected_new = np.zeros(n_people) # the people who are sick after this timestep

    for i_person in range(n_people):

        # if previously infected, still infected after this timestep
        if is_infected[i_person] == 1:
            is_infected_new[i_person] = 1

        # otherwise, each other person has some probability to infect this person
        # probabilities are given in p_spread
        else:
            for j_person in range(n_people):
                if j_person == i_person:
                    continue
                if np.random.random() < p_spread[i_person, j_person]:
                    is_infected_new[i_person] = 1
                    break
    return is_infected_new

# run an instance of this network spread model
# parameters are:
#   n_people: number of people
#   p_spread[i, j]: probability that person i will infect person j in one timestep, if person i is infected
#                   should be a symmetric matrix
#   p_init: probability that someone is infected in the 0th timestep (assume uniform across people for now)
#   n_steps: number of timesteps
def run_instance(n_people, p_init, n_steps):

    # model the probability person i spreads to person j in a given timestep as 1-e^(-\lambda * \tau)
    # \tau is the average time that i and j spend in "close" proximity in one timestep
    # this lambda is the scale factor
    lambda_param = 1.

    # random set of people are infected at 0th timestep
    is_infected = np.random.choice([0, 1], size=(n_people), replace=True, p=[1-p_init, p_init])

    # is_infected_accum[i, j]=1 if jth person was sick at the beginning of the ith timestep
    is_infected_accum = np.zeros((n_steps, n_people))
    is_infected_accum[0, :] = is_infected

    # if someone is infected, we need to run the steps
    if not np.sum(is_infected) == 0:
        for i_step in range(1, n_steps):
            p_spread = 1 - np.exp(-lambda_param * tau(i_step, n_people))
            is_infected_accum[i_step, :] = step(is_infected_accum[i_step-1, :], p_spread)

    return is_infected_accum

# run many instances of the network spread model
def run_model():

    n_people = 10
    n_steps = 100 # number of timesteps
    n_instances = 1000 # number of instances of the model to run to get an average

    # probability that someone is infected in the 0th timestep (assume uniform across people for now)
    # should set at whatever population believed to be for SB?
    p_init = 0.2 

    is_infected_accum_all_instance = np.zeros((n_steps, n_people))
    for i_instance in range(n_instances):
        is_infected_accum_all_instance += run_instance(n_people, p_init, n_steps)
    is_infected_accum_all_instance /= float(n_instances)

    # plot average trajectory of infection of person 0
    plt.plot(is_infected_accum_all_instance[:, 0])

    # attempted to derive analytical model to compare--doesn't match up right now
#     t_to_plot = np.arange(n_steps)
#     p_model = np.zeros(len(t_to_plot))
#     alpha = np.power(0.1, n_people-1)
# #     for i_t, t_to_plot_ind in enumerate(t_to_plot):
# #         p_model[i_t] = 1 - np.exp(np.log(alpha) * np.sum(np.power(n_people, np.arange(0, t_to_plot_ind))) + np.power(n_people, t_to_plot_ind) * np.log(1 - p_init))
#     for i_t, t_to_plot_ind in enumerate(t_to_plot):
#         if i_t == 0:
#             p_model[i_t] = p_init
#         else:
#             q = p_model[i_t - 1]
#             #p_model[i_t] = 1 - (1 - q) * np.power(1 - q * 0.1, n_people-1)
#             p_model[i_t] = q + (1 - q) * (1 - np.power(1 - q * 0.1, n_people-1))
#     plt.plot(t_to_plot, p_model)

    plt.show()

if __name__ == '__main__':

    run_model()
