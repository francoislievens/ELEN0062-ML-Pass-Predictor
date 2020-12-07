import numpy as np





x = np.arange(4)
y = np.arange(4)

z, a = np.meshgrid(x, y)
z = np.reshape(z, (-1, 1))
a = np.reshape(a, (-1, 1))
for i in range(0, len(z)):
    print('{} - {}'.format(z[i], a[i]))

b = np.zeros(z.shape)
for i in range(0, len(z)):
    b[i] = (z[i] + a[i])

b = np.reshape(b, (len(x), len(x)))
print(b)