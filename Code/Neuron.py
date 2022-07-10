import Coordinates


class Neuron():
    def __init__(self, coordinates, Dendrites, Axons, signal_modification, base_space):#, strength):
        super(Neuron, self).__init__()
        self.base_space = base_space
        self.coordinates = coordinates
        self.Dendrites = Dendrites # Incoming connections
        self.Axons = Axons # Outgoing connections
        self.name = ",".join([str(self.coordinates.x), str(self.coordinates.y), str(self.coordinates.z)])
#        self.strength = strength # how strong is the signal if the threshold is surpassed
        self.signal_modification = signal_modification # how does the neuron influence the signals it receives?
        self.active = False
        self.signal = 0
        self.fire_together = {}

    def activation(self, source, signal):
        self.signal = Coordinates.clamp(self.signal + signal, 0, 1.5)
        self.base_space.active_neurons[self.name] = self
        self.active = True
        for i in self.Axons:
            # TODO mehrere axone können die source sein
            if self.base_space.generate:
                i.receive_signal(self, self.signal_modification * signal)
        # send signal of strength x to all axons

    def step(self):
        if self.signal > 0:
            self.signal -= 0.33
        if self.signal <= 0:
            self.active = False
            self.signal = 0
            self.base_space.active_neurons.pop(self.name)

    def reset_for_next_run(self):
        self.signal = 0
        self.active = False
        self.fire_together = {}

    def color_me(self, color="black"):
        value = self.base_space.neuron_dot_dict[self.name]
        if value[1].active:
            value[0].set_color(color)