import numpy as np
import random
random.seed(1)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import Coordinates
import Perceptive_neuron, Processing_neuron, Interaction_neuron
import Connection


def new_positions_spherical_coordinates(self):
    phi = random.uniform(0, 2 * np.pi)
    costheta = random.uniform(-1, 1)
    u = random.uniform(0, 1)

    theta = np.arccos(costheta)
    r = ((size - 10) / 2) * np.sqrt(u)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return (x, y, z)


def new_positions_circular_coordinates(self):
    phi = random.uniform(0, 2 * np.pi)
    costheta = random.uniform(-1, 1)
    u = random.uniform(0, 1)

    size = 100
    theta = np.arccos(costheta)
    r = (size / 2) * np.sqrt(u)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    # z = r * np.cos(theta)
    return (x, y)


def ordered_input_neurons(self, height, width, plane_end):
    global size
    V = []
    area = size - 20
    y_distance = area / height
    z_distance = area / width
    Y = np.arange(-(size / 2) + 10, (size / 2) - 10, y_distance)
    Z = np.arange(-(size / 2) + 10, (size / 2) - 10, z_distance)
    for y in Y:
        for z in Z:
            V.append(Coordinates.Coordinate(plane_end, y, z))
    return V


def ordered_output_neurons(self, height, width, plane_end):
    global size
    V = []
    area = size - 20
    y_distance = area / height
    z_distance = area / width
    Y = np.arange(-(size / 2) + 10, (size / 2) - 10, y_distance)
    Z = np.arange(-(size / 2) + 10, (size / 2) - 10, z_distance)
    for y in Y:
        for z in Z:
            V.append(Coordinates.Coordinate(plane_end, y, 0))
    return V

def reach_out_to_previous_layer(self, layer, neuron, connection_number):
    available_neurons = {}
    for n in layer:
        available_neurons[Coordinates.distance_finder(neuron.coordinates, n.coordinates)] = n
    srtd = sorted(available_neurons.items())
    return [i[1] for i in srtd[:connection_number]]

def find_x_nearest(self, Neuron, setB, connection_limit=8, x=5): # finds x nearest Neurons of setB to Neuron
    distdict={}
    for i in setB:
        if i != Neuron and len(i.connections) < connection_limit and sum([(type(c.other_side(i)) == Perceptive_neuron.PerceptiveNeuron or type(c.other_side(i)) == Interaction_neuron.InteractionNeuron) for c in i.connections]) == 0:
            # check if neuron is perceptive and if i already connected to perceptive
            # this should ensure that one perceptive neuron does not connect to a processing neuron thats already connected to a perceptive neuron
            if type(Neuron) == Perceptive_neuron.PerceptiveNeuron:
                perceptive_connections = [(type(connections_of_i.other_side(connections_of_i)) == Perceptive_neuron.PerceptiveNeuron) for connections_of_i in i.connections]
                if sum(perceptive_connections) == 0:
                    distdict[Coordinates.distance_finder(Neuron.coordinates, i.coordinates)] = i
                # Debug output
#                else:
#                    print("prevented perceptives connecting to same neuron")
            else:
                distdict[Coordinates.distance_finder(Neuron.coordinates, i.coordinates)] = i
    srtd = sorted(distdict.items())
    return [i[1] for i in srtd[:x]]
