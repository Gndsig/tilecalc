#%%
import numpy as np
from shapely.geometry import Point

# %%
point = Point(1.0, 2.0)

# %%
print(point.area)
print(point.bounds)
print(point.length)
print(point.geom_type)
point

# %%
point = Point(0.0, 0.0)
polygon = point.buffer(10.0)

# %%


# %%


# %%


# %%


# %%
