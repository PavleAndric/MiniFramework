import numpy as np
np.random.seed(42)

tezina = np.array([[ 0.49671415 ,-0.1382643 ,  0.64768854],
 [ 1.52302986, -0.23415337 ,-0.23413696],
 [ 1.57921282 , 0.76743473, -0.46947439]])


pristars =  np.array([[ 0.54256004],
 [-0.46341769],
 [-0.46572975]])



out = np.array([[ 0.24196227],
 [-1.91328024],
 [-1.72491783]])


print(f" t_sahape {tezina.shape}, prisstast = {pristars.shape} , out  = {out.shape}")
X = np.dot(tezina, out) + pristars
print(X)