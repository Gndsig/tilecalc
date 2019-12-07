#%%
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Polygon
import shapely.wkt
import functools
import glob


#%%
line_sample1 = 'LINESTRING(139.07730102539062 36.00022956359412, 139.0814208984375 35.98022880246021)'
ls1 = shapely.wkt.loads(line_sample1)
ls2 = LineString([(139.07485444335938, 35.9923409624497), (139.07730102539062, 35.99356320664446),(139.07867431640625,35.99722880246021),(139.0824208984375,35.99842880246021)])


#%%
# sample
print(ls2.area)
print(ls2.bounds)
print(ls2.length)
print(ls2.geom_type)
print(ls2.distance)
list(ls2.coords)

#%%
lpi = LinkingPolylineImage(ls2)
bounds = lpi.terminal_node_aligned()
#bounds = lpi.xy_aligned(form='rectangle')
zoom=18


# %%
lpi.overlappingTileSegments(bounds, zoom=zoom)