# using masic command #%% as vscode like jupyter.
#%%
import os
import sys
#os.getcwd()
os.chdir("/workdir/linking_polyline_image/linking/")

from link import *

#%%
ls1 = 'LINESTRING(139.07650102539062 35.99722956359412, 139.0784208984375 35.995022880246021, 139.0814208984375 35.994022880246021)'
ls2 = LineString([(139.07585444335938, 35.9923409624497), (139.07730102539062, 35.99356320664446),(139.07867431640625,35.99722880246021),(139.0824208984375,35.99842880246021)])


#%%
# sample
ls1_ = shapely.wkt.loads(ls1)
print(ls1_.area)
print(ls1_.bounds)
print(ls1_.length)
print(ls1_.geom_type)
print(list(ls1_.coords))
ls1_

#%%
# sample
print(ls2.area)
print(ls2.bounds)
print(ls2.length)
print(ls2.geom_type)
print(list(ls2.coords))
ls2


#%%
lpi = LinkingPolylineImage(ls1)
zoom=18
filepath='../datasets'

bounds1 = lpi.xy_aligned(polyline=ls1, form='rectangle', minimum=[], minimum_unit='latlng', zoom=zoom)  # if not specify, class instance use.
bounds1_ = lpi.xy_aligned(form='minmax')
bounds2 = lpi.terminal_node_aligned(ls2)
#bounds2 = pi.terminal_node_aligned()  # if not specify, class instance use.
print(bounds1)
print(bounds1_)
print(bounds2)

# bounds form : [theta, np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]]) ]
# theta is rotation angle[rad] from x axis.
# xy_aligned only, form='minmax' as np.array([x_min,y_min],[x_max,y_max]) for using another function.
#%%
Polygon(bounds1[1])

#%%
Polygon(bounds2[1])

#%%
# file path is defined, return file path list.
pickup_tile_path = lpi.overlappingTiles(bounds1, zoom=zoom, filepath=filepath)
pickup_tile_path
#%%
# file path is not defined, return xy tile coordinate array.
pickup_tiles_ = lpi.overlappingTiles(bounds1, zoom=zoom, filepath=False)
pickup_tiles_

# %%
# file path is not defined, return xy tile coordinate array and where intersection is in tile.
pickup_tiles, pickup_tiles_intersection = lpi.overlappingTileSegments(bounds1, zoom=zoom, filepath=False)
print(pickup_tiles)
print(pickup_tiles_intersection)

# pickup_tiles_intersection as np.array([[x1, y1], [x2, y2],...]), 
# np.array([[x11,y11],[x12,y12],[x13,y13]], ...). x11, y11 and x12, y12 is intersections of bounds lines and tiles lines.
# x13, y13 is bounds corner points in the tiles. If intersections or corner points are not exist, it is in np.nan.
# elif filepath='anypath', return pickup_tiles, pickup_tiles_intersection, pickup_tiles_list. pickup_tile_list as ['filepath/zoom/x1/y1',...].

# %%
# file path is defined, return plus file path list.
pickup_tiles, pickup_tiles_intersection, pickup_tiles_path = lpi.overlappingTileSegments(bounds2, zoom=zoom, filepath=filepath)
print(pickup_tiles)
print(pickup_tiles_intersection)
print(pickup_tiles_path)
