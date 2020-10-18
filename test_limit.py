import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from scipy import stats

combined_max = .2

area_variation_low = 0.8
area_variation_high = 0.8

# MAX 1
overlap_variation_low = 1
overlap_variation_high = 1

overlaps = list()
areas = list()
unions = list()
intersections = list()


def calc_vals():
    av50 = 1
    if combined_max:
        variance = random.uniform(0, combined_max)
        overlap_var = 1 - variance
        remainder_variance = combined_max - variance
        # area_var = 1 - remainder_variance
        area_var = 1 + random.uniform(-remainder_variance, remainder_variance)
    else:
        area_var = random.uniform(area_variation_low, area_variation_high)
        overlap_var = random.uniform(overlap_variation_low, overlap_variation_high)

    vobs = av50 * area_var
    overlap = overlap_var
    intersection = min(av50*overlap, vobs)
    union = vobs - (intersection) + av50
    cci = intersection / union
    dci = (av50 - intersection) / av50

    overlaps.append(overlap)
    areas.append(vobs)
    unions.append(union)
    intersections.append(intersection)
    return cci, dci

y = list()
x = list()
for _ in range(5000):
    cci, dci = calc_vals()
    y.append(cci)
    x.append(dci)

def make_colors(data):
    norm = max(data)
    normed = [number/norm for number in data]
    graph_colors = [cm.jet(x) for x in normed]
    m = cm.ScalarMappable(cmap=cm.jet, norm=colors.Normalize(vmax=norm))
    m.set_array(graph_colors)
    return m, graph_colors

xs = np.linspace(0, 10, 1000)

# p = np.polyfit([0.818, 0.86, 0.9], [0.1, 0.0976, 0.1], 2) #0.2
# p = np.polyfit([0.53846, 0.6455, 0.7], [0.3, 0.2777, 0.3], 4) #0.3
p = np.polyfit([0.1, 0.9], [0., 0.9], 1) #0.2

x_ind = np.array(x).argsort()
x = np.array(x)[x_ind]
y = np.array(y)[x_ind]

areas = np.array(areas)[x_ind]
unions = np.array(unions)[x_ind]
intersections = np.array(intersections)[x_ind]

p = np.polyfit(y, x, 2)
p_f = np.poly1d(p)

def make_graph(num, data):
    plt.subplot(f'21{num}')
    plt.xlim(0, 0.5)
    plt.xlabel('DCI')
    plt.ylim(0.5, 1)
    plt.ylabel('CCI')
    bar, graph_colors = make_colors(data)
    plt.scatter(x, y, color=graph_colors, s=0.1)
    plt.colorbar(bar)
    # plt.plot
    limit = 0.3
    lower_bound = 1/(1+limit)
    plt.plot(xs, lower_bound - lower_bound * xs)  #  +10% bigger
    # plt.plot(xs, 1 - xs)  #  same size or smaller
    plt.plot(xs, 1*xs)


plt.suptitle(f'Area +/- {area_variation_low * 100}% Overlap: >= {(1-overlap_variation_low)*100}%')
make_graph(1, unions)
make_graph(2, intersections)
# plt.scatter(x, intersections)
plt.show()
