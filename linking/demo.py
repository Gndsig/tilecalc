# using masic command #%% as vscode like jupyter.

# 長方形をpixelの方で作るためのデバッグ

#%%
import os
import sys
#os.getcwd()
os.chdir("/workdir/linking_polyline_image/linking/")

from link import *


ls1 = 'LINESTRING(139.07650102539062 35.99722956359412, 139.0784208984375 35.995022880246021, 139.0814208984375 35.994022880246021)'
ls2 = LineString([(139.07685444335938, 35.9923409624497), (139.07730102539062, 35.99356320664446),(139.07797431640625,35.99622880246021),(139.0804208984375,35.99772880246021)])


#polyline = ls1
#polyline = shapely.wkt.loads(polyline)

polyline = ls2


#%%
lpi = LinkingPolylineImage()
zoom=18

bounds1 = lpi.xy_aligned(polyline=polyline, form='rectangle', minimum=[], unit=['latlng','pixel'], zoom=zoom)  # if not specify, class instance use.
bounds1_ = lpi.xy_aligned(polyline=polyline, form='minmax')
bounds2 = lpi.terminal_node_aligned(polyline, unit=['latlng','pixel'], buff=[100,100])
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

#bounds = bounds1.copy()
bounds = bounds2.copy()

# return xy tile coordinate array.
#pickup_tiles = lpi.overlappingTiles(bounds, zoom=zoom)
#pickup_tiles

# %%
# return xy tile coordinate array and where intersection is in tile.
pickup_tiles, pickup_tiles_intersection = lpi.overlappingTileSegments(bounds, zoom=zoom)
print(pickup_tiles)
print(pickup_tiles_intersection)

# pickup_tiles_intersection as np.array([[x1, y1], [x2, y2],...]), 
# np.array([[x11,y11],[x12,y12],[x13,y13]], ...). x11, y11 and x12, y12 is intersections of bounds lines and tiles lines.
# x13, y13 is bounds corner points in the tiles. If intersections or corner points are not exist, it is in np.nan.
# elif filepath='anypath', return pickup_tiles, pickup_tiles_intersection, pickup_tiles_list. pickup_tile_list as ['filepath/zoom/x1/y1',...].
#%%
lpi.concat_tile(bounds, pickup_tiles, \
        file_path='../datasets/', file_extention='.webp', \
        save_path='./', draw_polyline=ls2, draw_bounds=True, crop_mode='dst')

#%%
lpi.concat_tile_segment(pickup_tiles, pickup_tiles_intersection, \
        file_path='../datasets/', file_extention='.webp', \
        save_path='./', draw_polyline=ls2, draw_bounds=True, crop_mode='dst')