import time

import numpy as np
import random
random.seed(1)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import Coordinates
import Neuron
import Connection
import independent_functions
from scipy.stats import norm
import scipy.stats


from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

size = 100
class NeuronSpace():
    def __init__(self, Visualization):
        super(NeuronSpace, self).__init__()
        self.iter = 0
        self.ticks = 0
        self.number_of_dendrites = 4 # this is equal to the window size
        self.generate = False
        self.Visualization = Visualization
        global size
        self.size = size
        self.spawn_neurons_axons()


    def start_vis(self):
        plt.ion()
        self.neuron_dot_dict = {}  # name: (neuron, punkt auf plot)
        self.axon_line_dict = {}  # name: (axon, linie auf plot)
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-(size / 2), size / 2)
        self.ax.set_ylim(-(size / 2), size / 2)
        self.ax.set_zlim(-(size / 2), size / 2)

        for layer in self.layers:  # plot all neurons
            for neuron in layer:
                self.neuron_dot_dict[neuron.name] = [(self.ax.scatter(neuron.coordinates.x, neuron.coordinates.y, neuron.coordinates.z, c="grey",
                                                             s=10 * neuron.signal_modification)), neuron]

        for a in self.Synapse_dict.values():
            self.axon_line_dict[a.name] = [(self.ax.plot3D([a.Axon_side.coordinates.x, a.Dendrite_side.coordinates.x],
                                                           [a.Axon_side.coordinates.y, a.Dendrite_side.coordinates.y],
                                                           [a.Axon_side.coordinates.z, a.Dendrite_side.coordinates.z], linewidth=1,
                                                           c='grey')), a]

        self.grown_axons = []

    def create_Axon(self, axon_side, dendrite_side):
        in_name_v_first = True
        if axon_side.coordinates.x < dendrite_side.coordinates.x:
            in_name_v_first = False
        elif axon_side.coordinates.x == dendrite_side.coordinates.x:
            if axon_side.coordinates.y < dendrite_side.coordinates.y:
                in_name_v_first = False
            elif axon_side.coordinates.y == dendrite_side.coordinates.y:
                if axon_side.coordinates.z < dendrite_side.coordinates.z:
                    in_name_v_first = False
        name = ""
        if in_name_v_first:
            name = axon_side.name + "," + dendrite_side.name
        else:
            name = dendrite_side.name + "," + axon_side.name
        if name not in self.Synapse_dict.keys():
            synapse = Connection.Connection(axon_side, dendrite_side, name=name, size=size, base_space=self, threshold=0.5 * random.uniform(0.7, 1.3))
            self.Synapse_dict[name] = synapse
            axon_side.Axons.append(synapse)
            dendrite_side.Dendrites.append(synapse)
            return synapse

    def colour_markov_blanket(self, neuron):
        neuron.my_markov_blanket()

        (self.neuron_dot_dict[neuron.name])[0].set_color("purple")

        for p in neuron.parents:  # parents are sensory states in green
            tpl = self.neuron_dot_dict[p.name]
            tpl[0].set_color("green")

        for c in neuron.children:   # children are active states in blue
            tpl = self.neuron_dot_dict[c.name]
            tpl[0].set_color("blue")

        for pc in neuron.parents_of_children:   # parents of children are active states in blue
            tpl = self.neuron_dot_dict[pc.name]
            tpl[0].set_color("blue")

        print("Chosen neuron in purple\nparents are sensory states in green\nchildren are active states in blue\nparents of children are active states in blue")

    def draw_brain(self, active_axons):
        # visualize the neurons
        for key in self.neuron_dot_dict:
            value = self.neuron_dot_dict[key]
            if value[1].active:
                value[0].set_color("red")
            else:
                value[0].set_color("grey")
            value[0].set_sizes([50 * value[1].signal_modification])

        for key in self.axon_line_dict:
            value = self.axon_line_dict[key]
            if value[1].active:
                value[0][0].set_color("red")
            else:
                value[0][0].set_color("grey")
        for n in self.grown_axons:
            value = self.axon_line_dict[n]
            value[0][0].set_color("green")
        if len(self.grown_axons) > 0:
            print("strengthened ", len(self.grown_axons), " axons")
        for n in self.new_axons:
            value = self.axon_line_dict[n.name]
            value[0][0].set_color("purple")
        if len(self.new_axons) > 0:
            print("grew ", len(self.new_axons), " axons")
#        self.fig.savefig('..//Bilder//temp'+str(self.ticks)+'.png', dpi=self.fig.dpi)
        self.grown_axons=[]
        self.new_axons = []

    def Hebbian(self):

        for n in self.active_neurons:
            if n in self.PNeuron_dict.keys():
                n = self.PNeuron_dict[n]
                for z in self.active_neurons:
                    if z in self.PNeuron_dict.keys():
                        z = self.PNeuron_dict[z]
                        if n != z:
                            if z not in [o.other_side(own=n) for o in n.connections] and z.name not in n.fire_together.keys():
                                n.fire_together[z.name] = 2
                            if z.name in n.fire_together.keys():
                                if n.fire_together[z.name] > 30:
                                    a = self.create_Axon(n, z)
                                    if a != None:
                                        self.new_axons.append(a)
                                        #print("Axon grown!")
                                        if self.Visualization:
                                            self.axon_line_dict[a.name] = [
                                                (self.ax.plot3D([a.Axon_side.coordinates.x, a.Dendrite_side.coordinates.x],
                                                                [a.Axon_side.coordinates.y, a.Dendrite_side.coordinates.y],
                                                                [a.Axon_side.coordinates.z, a.Dendrite_side.coordinates.z], linewidth=1,
                                                                c='grey')), a]
                                    n.fire_together = {}
                                elif z.name in n.fire_together.keys():
                                    n.fire_together[z.name] += 3

    def run(self):
        if self.learn:
            for i in self.Pset:
                for f in list(i.fire_together):
                    i.fire_together[f] -= 1
                    if i.fire_together[f] == 0:
                        i.fire_together.pop(f)
            self.Hebbian()


        aavalues = list(self.active_axons.values())
        for i in aavalues:
            i.step()
        anvalues = list(self.active_neurons.values())
        for i in anvalues:
            i.step()

        #hier gehts weiter


    def spawn_neurons_axons(self):

        Layer_coordinates = []
        # choose coordinates for perceptive neurons, set V
        # these should face one plane of the neuron space
        Layer_coordinates.append(independent_functions.ordered_input_neurons(height = 8, width = 8, plane_end=-(size/2), size = self.size))


        # choose cluster of coordinates in the middle of the neuron space for processing neurons, set P
        Layer_coordinates.append(independent_functions.ordered_input_neurons(height = 8, width = 8, plane_end=-(size/2)+20, size = self.size))
        Layer_coordinates.append(independent_functions.ordered_input_neurons(height = 4, width = 4, plane_end=-(size/2)+40, size = self.size))
        Layer_coordinates.append(independent_functions.ordered_input_neurons(height = 2, width = 2, plane_end=-(size / 2) + 60, size = self.size))

        # choose cluster of coordinates on plane, opposite side to V, set I
        # that only connect to processing neurons
        Layer_coordinates.append(independent_functions.ordered_input_neurons(height=10, width=1, plane_end=size/2, size = self.size))


        # Neuron generation
        self.Neuron_dict = {}
        self.layers = []

        for l in Layer_coordinates:
            Layer = []
            for n in l:
                new_neuron = Neuron.Neuron(n, [], [], 1, base_space = self)
                self.Neuron_dict[new_neuron.name] = new_neuron
                Layer.append(new_neuron)
            self.layers.append(Layer)

        # spawn a bunch of Perceptive neurons on coordinate set V
        self.Vset = self.layers[0]
        # spawn a bunch of Processing neurons on coordinate set P
        self.Pset = self.layers[1:-1]
        # spawn a bunch of Interaction neurons on coordinate set I
        self.Iset = self.layers[-1]


        self.Synapse_dict = {}

        for idx, l in enumerate(self.layers):
            if idx != 0:
                for n in l:
                    chosen_neurons = independent_functions.reach_out_to_previous_layer(self.layers[idx-1], n, self.number_of_dendrites)
                    for chosen_neuron in chosen_neurons:
                        self.create_Axon(chosen_neuron, n)

        # build the markov blanket for every neuron:
        for l in self.layers:
            for n in l:
                n.my_markov_blanket()
#                n.Prior = norm.beta(1, 1) # Initialized without any prior beliefs
                # Every neuron only has one axon so this only gets one prior to send through the axon

        self.grown_axons = []
        self.new_axons = []
        if self.Visualization:
            self.start_vis()
            self.draw_brain(active_axons={})

    def feed_image_unsupervised(self, datapoint, target, learn):
        self.learn = learn
        self.stop = False # this becomes true once an interactive neuron gets activated
        if self.learn:
            print("Learning")
        else:
            print("not learning")
        self.generate = False
        self.active_neurons = {}
        self.active_axons = {}

        for v, idx in zip(self.Vset, np.arange(len(self.Vset))):
            v.activation(source="input", signal = datapoint[idx])
            # TODO
            # das wieder rein nehmen, damit vorhersage mit output abgeglichen werden kann
            #self.Iset[target].activation(1)
        training_runs = 0
#        self.run()
#        self.ticks += 1
        if self.Visualization:
            self.draw_brain(self.active_axons)

        # if no input neurons or processing neurons or axons are active
        while len(self.active_neurons.keys()) + len(self.active_axons.keys()) > 0 or training_runs==0:
            self.grown_axons = []
            self.new_axons = []
            self.run()
            self.ticks += 1
            print("Tick: ", self.ticks)
            if self.Visualization:
                self.draw_brain(self.active_axons)
            training_runs += 1
            if self.stop:
                break
            # solange irgend ein neuron noch an is:
            # die neuronen sind länger an, senden aber nur am anfang ein signal an die axone

        if self.learn:
            print("Decay axons")
            self.to_remove = []
            for all_axons in self.Synapse_dict.values():
                all_axons.forget()

            # only one is deleted to avoid mass extinction
            if len(self.to_remove) >= 1:
                self.Synapse_dict.pop(self.to_remove[0].name)
                if self.Visualization:
                    self.axon_line_dict.pop(self.to_remove[0].name)
                self.to_remove[0].Axon_side.connections.remove(self.to_remove[0])
                self.to_remove[0].Dendrite_side.connections.remove(self.to_remove[0])
                del self.to_remove[0]

            if len(self.to_remove) > 0:

                print(len(self.to_remove), " axons at max threshold")


        # clean up for next run
        self.active_neurons = {}
        self.active_axons = {}
        self.stop = False
        for ad in self.Synapse_dict:
            self.Synapse_dict[ad].last_signal = self.ticks - self.Synapse_dict[ad].cooldown - 1
            self.Synapse_dict[ad].reset_for_next_run()
        for Nd in self.Neuron_dict:
            self.Neuron_dict[Nd].reset_for_next_run()

        print("Picture processed")
        # self.start_vis()
        # self.draw_brain([])

    def generate_image(self, datapoint, target):
        self.generate = True
        self.active_neurons = {}
        self.active_axons = {}

        self.Iset[target].activation("input", 1)

        self.run()
        self.ticks += 1
        if self.Visualization:
            self.draw_brain(self.active_axons)
        while len(self.active_neurons.keys()) + len(self.active_axons.keys()) != 0:
            self.run()
            for all_axons in self.Synapse_dict.values():
                all_axons.forget()
            self.ticks += 1
            print("Tick: ", self.ticks)
            if self.Visualization:
                self.draw_brain(self.active_axons)
            # solange irgend ein neuron noch an is:
            # die neuronen sind länger an, senden aber nur am anfang ein signal an die axone
        print("Bild generiert")


neuronspace = NeuronSpace(Visualization = True)
neuronspace.colour_markov_blanket((neuronspace.layers[1])[28])
from sklearn import datasets
from sklearn import preprocessing
import pandas as pd
min_max_scaler = preprocessing.MinMaxScaler()
#training data
df = pd.DataFrame(datasets.load_digits().data)
#fit and transform the training data and use them for the model training
df1 = min_max_scaler.fit_transform(df)
digits = datasets.load_digits()
images = 0

print("Sensing images \n"
      "Symptoms: \n"
      "-Quick decay of axons \n"
      "-Activation function is the binary step function to simulate synapse transmitting signals \n"
      "-Not emergent")

for d,t in zip(df1, digits.target):
    #neuronspace.generate_image(d, t)
    print("durchgang ", images, " start tick: ", neuronspace.ticks)
    neuronspace.feed_image_unsupervised(d, t, learn = True)
    images += 1
    if images == 20:
        print("stop")
    if images == 10:
        print("stop")
        neuronspace.Visualization = True
        plt.ion()
        neuronspace.start_vis()


print("check")