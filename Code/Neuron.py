import Coordinates
import scipy.stats
from scipy.stats import beta
import numpy as np

class Neuron():
    def __init__(self, coordinates, Dendrites, Axons, base_space):
        super(Neuron, self).__init__()
        self.base_space = base_space
        self.coordinates = coordinates
        self.Dendrites = Dendrites # Incoming connections
        self.Axons = Axons # Outgoing connections
        self.name = ",".join([str(self.coordinates.x), str(self.coordinates.y), str(self.coordinates.z)])
        self.fire_together = {}
        self.list_to_generate_beta=[1]
        self.list_gen_beta_b = [1]
        self.Prior = beta.stats(sum(self.list_to_generate_beta),1)
        self.N = 101
        self.x, self.step = np.linspace(0, 1, self.N, retstep=True)


    def activation(self, Likelihood):
        #self.list_to_generate_beta
        self.likelihoods_a.extend([l for l in Likelihood])
        #self.list_gen_beta_b
        self.likelihoods_b.extend([1-l for l in Likelihood])

    def step(self):
        Prior_pdf = beta.pdf(self.x, self.list_to_generate_beta, self.list_gen_beta_b)
        Prior_index = np.where(np.max(Prior_pdf) == Prior_pdf) # wahrscheinlichkeit ist hier am groeßten
        Prior_probab = Prior_pdf[Prior_index]*self.step


        Likelihood_pdf = beta.pdf(self.x, self.list_to_generate_beta, self.list_gen_beta_b)
        Likelihood_index = np.where(np.max(Likelihood_pdf) == Likelihood_pdf) # wahrscheinlichkeit ist hier am groeßten
        Likelihood_probab = Likelihood_pdf[Likelihood_index]*self.step



        posterior = beta(sum(self.list_to_generate_beta), sum(self.list_gen_beta_b))
        for i in self.Axons:
            i.receive_signal(self, posterior)
        # send posterior to all axons

        self.list_to_generate_beta = [1]
        self.list_gen_beta_b = [1]

    def reset_for_next_run(self):
        self.signal = 0
        self.active = False
        self.fire_together = {}

    def color_me(self, color="black"):
        value = self.base_space.neuron_dot_dict[self.name]
        if value[1].active:
            value[0].set_color(color)

    def my_markov_blanket(self):
        # Parents aka sensory states
        self.parents = [d.Axon_side for d in self.Dendrites]
        # Children aka active states
        self.children = [c.Dendrite_side for c in self.Axons]
        # Parents of children aka active states
        self.parents_of_children = ([[dend.Axon_side for dend in kid.Dendrites if dend.Axon_side != self] for kid in children])[0]
        self.sensory_states = self.parents
        self.active_states = self.children + self.parents_of_children
        self.internal
        return self.parents, self.children, self.parents_of_children
