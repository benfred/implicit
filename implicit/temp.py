# finding largest/smallest k
import numpy as np


# largest k's
array = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 10])

indices = np.argpartition(array, -3)[-3:]
values = array[indices]

print(indices), print(values)

# smallest k's
array = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 10])

indices = np.argpartition(array, 3)[:3]
values = array[indices]

print(indices), print(values)

