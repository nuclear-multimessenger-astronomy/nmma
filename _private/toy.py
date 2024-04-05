import numpy as np
a= [[1,3],[2,4],[7,3],[2,8]]
b= [0,2,1,0,3,0,2,0,1,2,0,2,1,0,1]
a=np.array(a)
# b=np.array(b)
# print(a[b])



a= np.array([[[1,3],[2,4],[7,3],[2,8]], [[1,3],[2,4],[7,3],[2,8]]])
for i, data in enumerate(a):
    rad, mass, rho, lam = data
    print(i, data, mass[-1], lam[-1])
print(a[:, 0, -1], a[:, 1, -1])

