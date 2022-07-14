import random
import numpy as np
import Coordinates
import math


class Connection():
    # simulates the axon and its synapse
    def __init__(self, Axon_side, Dendrite_side, name, size, base_space, threshold = 0.65):
        super(Connection, self).__init__()
        self.base_space = base_space

        self.Axon_side = Axon_side
        self.Dendrite_side = Dendrite_side
        self.name = name

        self.weight=1

    def receive_signal(self, distribution):
        rest = self.weight%1
        draws = math.floor(self.weight)
        if rest != 0:
            samples = distribution.rvs(draws + 1)
            samples[-1] = samples[-1]*rest
        else:
            samples = distribution.rvs(draws)
        self.Axon_side.activation(self.Dendrite_side, samples)


    def strengthen(self):
        self.threshold = self.threshold * 0.95
        #print(self.base_space.ticks, " axon grew threshold")
        self.base_space.grown_axons.append(self.name)

    def forget(self):
        # The Axon should not disappear if either of its adjacent Processing neurons only has two Axons
        self.threshold = Coordinates.clamp(self.threshold*1.05, 0.02,0.8)
        # Axon disappears because it was not often used
        if not self.relevant_axon:
            if self.threshold == 0.8:
                if len(self.Axon_side.connections) > 2 and len(self.Dendrite_side.connections) > 2:
                    self.base_space.to_remove.append(self)

    def randomize(self):
        self.threshold = self.threshold * random.uniform(0.8,1.2)

    def other_side(self, own):
        if own == self.Axon_side:
            return self.Dendrite_side
        return self.Axon_side

    def reset_for_next_run(self):
        self.active = False

    def color_me(self, color="black"):
        value = self.base_space.axon_line_dict[self.name]
        if value[1].active:
            value[0][0].set_color(color)