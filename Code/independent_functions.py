import numpy as np
import random
random.seed(1)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import Coordinates
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


def ordered_input_neurons(height, width, plane_end, size):
    V = []
    area = size - 20
    y_distance = area / height
    z_distance = area / width
    Y = np.arange(-(size / 2) + 10, (size / 2) - 10, y_distance)
    Z = np.arange(-(size / 2) + 10, (size / 2) - 10, z_distance)
    for y in Y:
        if height == 1:
            y = 0
        for z in Z:
            if width == 1:
                z = 0
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

def reach_out_to_previous_layer(layer, neuron, connection_number):
    available_neurons = {}
    for n in layer:
        available_neurons[Coordinates.distance_finder(neuron.coordinates, n.coordinates)] = n
    srtd = sorted(available_neurons.items())
    return [i[1] for i in srtd[:connection_number]]


'''
combine two distributions in matlab:

sigma_p = 1;
v_p = 3;
sigma_u = 1;
u = 2;
MINV = 0.01;
DV = 0.01;
MAXV = 5;
vrange = [MINV:DV:MAXV];
numerator = normpdf(vrange,v_p,sigma_p) .* normpdf (u,vrange.^2,sigma_u);
normalization = sum(numerator*DV);
p = numerator/normalization;
plot(vrange,normpdf(vrange,v_p,sigma_p),'k');
figure()
plot(vrange,normpdf(u,vrange.^2,sigma_u),'k');
figure()
plot(vrange,p,'k');
xlabel("v");
ylabel("p(v|u)");


'''

