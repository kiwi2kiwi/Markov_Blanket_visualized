import Coordinates

class InteractionNeuron():
    def __init__(self, coordinates, connections, signal_modification, base_space, id, name="empty"):#, strength):
        super(InteractionNeuron, self).__init__()
        self.base_space = base_space
        self.coordinates = coordinates
        self.connections = connections
        self.name = ",".join([str(self.coordinates.x), str(self.coordinates.y), str(self.coordinates.z)])
#        self.strength = strength # how strong is the signal if the threshold is surpassed
        self.signal_modification = signal_modification # how does the neuron influence the signals it receives?
        self.active = False
        self.signal = 0
        self.id = id

    def activation(self, source, signal):

        # self.base_space.active_neurons[self.name] = self # ignore
        if not self.base_space.generate:
            self.base_space.stop = True
            self.active = True
            print("Environment interaction neuron ",self.id," outputs: ", (self.signal*self.signal_modification))
        else:
            self.base_space.active_neurons[self.name] = self  # ignore
            self.active = True
            self.signal = Coordinates.clamp(self.signal + signal, 0, 1.5)
            for i in self.connections:
                if i != source:
                    i.receive_signal(self, self.signal_modification * signal)
        # output signal

    def step(self):
        if self.signal > 0:
            if not self.base_space.generate:
                self.signal -= 0
                self.active = True  # Damit das Programm zu ende is
                #self.base_space.active_neurons.pop(self.name)
            else:
                self.signal -= 0.1
                self.active = True
        if self.signal <= 0:
            if self.base_space.generate:
                self.active = False
                self.base_space.active_neurons.pop(self.name)
            self.active = False
            self.signal = 0

    def reset_for_next_run(self):
        self.signal = 0
        self.active = False

    def color_me(self, color="black"):
        value = self.base_space.neuron_dot_dict[self.name]
        if value[1].active:
            value[0].set_color(color)