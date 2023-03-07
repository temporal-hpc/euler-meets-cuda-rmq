import numpy as np
import pandas as pd
import sys

# FUNCTIONS
def sigmoid(x, s=0.8, k=0.1):
    # sigmoid function
    # use k to adjust the slope
    s = 1 / (1 + np.exp( (-x + s) / k))
    return s

if len(sys.argv) != 6:
    print(f"Run as:\n\tpython {sys.argv[0]} <csv_path> <title> <s> <k> <a>\n")
    print("s : sigmoid threshold value [0,1]")
    print("k : threshold slope strength (smaller is stronger)")
    print("a : overall transparency [0: transparent, 1: solid]\n")
    exit()

def get3Ddata(file):
    hc = pd.read_csv(file)
    #hc = hc.groupby(['dev', 'alg','reps','n','bs','q','lr']).agg([np.mean, np.std]).reset_index();
    hc = hc.groupby(['dev', 'alg','n','bs','q','lr'],sort=False).agg([np.mean, np.std]).reset_index();
    #print("HC",hc['n'].min())
    hc['n-exp'] = np.log2(hc['n'])
    hc['nb'] = np.log2(hc['n'] / hc['bs'])
    #print("nb",hc['nb'])
    hc['lr-ratio'] = np.round(np.log2(hc['lr'] / hc['n']))
    hc['mean_ns/q'] = hc['ns/q','mean']
    return hc[['n-exp', 'nb', 'lr-ratio', 'mean_ns/q']].to_numpy()




# SCRIPT

data_path = sys.argv[1]
title     = sys.argv[2]
sigmoid_s = float(sys.argv[3])
sigmoid_k = float(sys.argv[4])
alpha = float(sys.argv[5])
print(f"ARGS:\n{data_path=}\n{sigmoid_s=}   {sigmoid_k=}   {alpha=}\n")
## Load data using pandas
#full = np.genfromtxt(data_path, delimiter=",")
full = get3Ddata(data_path)
#print(full[:,0])
## Get the axis span
x_nb = np.unique(full[:, 1])
y_n = np.unique(full[:, 0])
z_lr = np.unique(full[:, 2])

print("X nb:", x_nb)
print("Y n:", y_n)
print("Z lr:", z_lr)

# NUEVO 3D
## Create the data matrix for easier data handling
times = np.ones((len(x_nb)* len(y_n)* len(z_lr)))*np.max(full[:,-1])
times = times.reshape(len(x_nb),len(y_n),len(z_lr))
print("times: ",times.shape)
print(f"n=[{y_n.min()}..{y_n.max()}]    nb=[{x_nb.min()}..{x_nb.max()}]    lr=[{z_lr.min()}..{z_lr.max()}]\n")

## Load the actual times
for i, tup in enumerate(full):
    # num blocks
    ax_nb = int(tup[1]) - int(x_nb.min())
    # n
    ax_n = int(tup[0]) - int(y_n.min())
    # lr
    ax_lr = int(tup[2]) - int(z_lr.min())
    t = tup[3]

    #print(f"{ax_nb=}, {ax_n=}, {ax_lr=}")
    #input()
    #print(f"{tup=}")
    #print(f"nb={2**(ax_n+1)/2**(ax_nb)}  n={2**int(tup[0])}  lr={2**(ax_n+1) * 2**int(tup[2])}  --> {t=}")
    #input()
    times[ax_nb,ax_n,ax_lr] = t
    #times[ax_lr,ax_nb,ax_n] = t

#times[3,0,0] = 0

#print("listo")
## Transform data from 1D -> 3D
#times = times.transpose(1,0,2)






#normalizar por 'n'
for i in range(times.shape[1]):
    slice = times[:,i,:].copy()
    slice_normalized = (slice - np.min(slice)) / (np.max(slice) - np.min(slice))
    times[:, i, :] = slice_normalized

# REVISAR
#times = np.flip(times, axis=1)
#times = np.flip(times, axis=0)

## Repeat each data point <rep> times to make each pixel larger
## Trick to improve visualization
rep = 20
times = np.repeat(times, rep, axis=0)
times = np.repeat(times, rep, axis=1)
times = np.repeat(times, rep, axis=2)

## Actual plot
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Extract the color map function
import matplotlib
cmap = matplotlib.cm.get_cmap('viridis')


data = times
## Color array
d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
## Alpha
# Option 1: Fixed alpha
#d2[..., 3] = 4

# Option 2: Data dependent
strength = 30  # <-- Filters out lower values, a larger value is a more agressive filter

# polinomio
#d2[..., 3] = np.power(1 - data, strength) * (255.)
d2[..., 3] = np.maximum(0.00,sigmoid(1 - data, s=sigmoid_s, k=sigmoid_k)) * alpha*(255.)

## Color values
# Filter helps to distribute the color map values among the values that are shown
# (with significant alpha)
# Renormalize filtered values to [0, 1] (the 10th percentile)
for i in range(data.shape[1]):
    filter = np.percentile(data[:,i,:], 10.0) # Divide the cmap among the 10th percentile of values
    data[:,i,:][data[:,i,:] > filter] = filter # Filter the rest
    slice = data[:,i,:].copy()
    slice_normalized = (slice - np.min(slice)) / (np.max(slice) - np.min(slice))
    data[:, i, :,] = slice_normalized

# Option 1: Scale
d2[..., 0] = cmap(data)[...,0] * 255#0#255
d2[..., 1] = cmap(data)[...,1] * 255#0#255
d2[..., 2] = cmap(data)[...,2] * 255#0#255

# Option 2: Fixed color
#d2[..., 0] = 0x00
#d2[..., 1] = 0x99
#d2[..., 2] = 0xff


## Uncomment to paint the axes
#d2[:, 0, 0] = [255,0,0,255]
#d2[0, :, 0] = [0,255,0,255]
#d2[0, 0, :] = [0,0,255,255]


"""
Demonstrates GLVolumeItem for displaying volumetric data.

"""
from pyqtgraph.Qt import QtCore, QtGui,QtWidgets

## Background color
pg.setConfigOption('background', (255,255,255,255))


app = QtWidgets.QApplication([])
w = gl.GLViewWidget()
# Switch to 'nearly' orthographic projection.
w.opts['distance'] = 2000
w.opts['fov'] = 1
w.show()
w.setWindowTitle(f'RTXRMQ Heat Cube: {title}')


## Bottom grid
g = gl.GLGridItem(glOptions='opaque', color=(0,0,0,64))
#g.translate(50,50,0)
g.scale(20, 20, 1)
g.setDepthValue(2)
w.addItem(g)

## Volume data
v = gl.GLVolumeItem(d2)
v.scale(0.5,0.5,0.5)
v.setDepthValue(2)
w.addItem(v)


## Text
def genText(pos, color, text):
    t = gl.GLTextItem()
    t.setData(pos=pos, color=color, text=text)
    t.scale(10,10,10)
    return t
#                  x, y, z
w.addItem(genText((2, 0, 0), (255, 0, 0, 255), "# blocks"))
w.addItem(genText((0, 2, 0), (0, 255, 0, 255), "n"))
w.addItem(genText((0, 0, 2), (0, 0, 255, 255), "lr"))


## Axis lines, not very visible
ax = gl.GLAxisItem(glOptions='opaque')
ax.setSize(x=10, y=10,z=10)
ax.scale(10,10,10)
w.addItem(ax)

if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtWidgets.QApplication.instance().exec_()

