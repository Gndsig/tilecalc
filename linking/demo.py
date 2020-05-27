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


#polyline = shapely.wkt.loads(ls1)
polyline = ls2

#%%
lpi = LinkingPolylineImage()
zoom=18


#def xy_aligned(self, polyline, minimum=[], buff=[], unit=['latlng','pixel'], form='rectangle', zoom=18):
a=  """function to Minimum Bounding Rectangle aligned xy axis.
    polyline (shapely.geometry.LineString) : LineString object. If default, use class instance.

    minimum (list of [x_min,y_min]) : Minimum width of Minimum Bounding Rectangle. Unit follows output unit.
    buff (list of [x_min, y_min]) : A rectangle buffer. Unit follows output unit.

    unit (list of str) [['latlng', 'pixel'], ['pixel', 'pixel'] ] : The first in the list shows the unit of input.
    THe second in the list shows the unit of output. And minimum unit and buff unit follows the second in the list.
    If unit is ['latlng','pixel'], the inputed polyline unit is longtude and latitude, and return unit is pixel and minimum unit is pixel, 

    form (str)['minmax', 'rectangle'] : If 'minmax', return np.array([[x_min, y_min], [x_max, y_max]]).
    If 'rectangle', return [theta(=0), np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]]) ]
    Theta is angle[rad] from x axis, this case is 0.

    Finaly, the unit of minimum and return should be unified.
    Warning, If unit is "latlng", later convert "latlng" to "pixel", point of rectangle is a little shift, no more parallelograms.

    If width < minimum, width=minimum and Rectangle is helf width from center point.
    """

bounds1 = lpi.xy_aligned(polyline, minimum=[], buff=[], \
                unit=['latlng','pixel'], form='rectangle', zoom=zoom)  # if not specify, class instance use.

bounds2 = lpi.terminal_node_aligned(polyline, unit=['latlng','pixel'], buff=[100,100])
#bounds2 = pi.terminal_node_aligned()  # if not specify, class instance use.
print(bounds1)
print(bounds2)

# bounds form : [theta, np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]]) ]
# theta is rotation angle[rad] from x axis.
# xy_aligned only, form='minmax' as np.array([x_min,y_min],[x_max,y_max]) for using another function.
#%%
Polygon(bounds1[1])

#%%
Polygon(bounds2[1])

#%%

bounds = bounds1.copy()
#bounds = bounds2.copy()

# return xy tile coordinate array.
#def overlappingTiles(self, bounds, unit=['pixel','pixel'], zoom='self', TILE_SIZE='self'):
a=  """
    bounds (list) : input [theta, np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]])] by xy_aligned or terminal node aligned.
    zoom (int)[0-18] : zoom level.
    
    unit (list of str) [['latlng', 'pixel'], ['pixel', 'pixel'] ] : The first in the list shows the unit of input.
    THe second in the list shows the unit of output. And minimum unit follows the second in the list.
    If unit is ['latlng','pixel'], the inputed polyline unit is longtude and latitude, and return unit is pixel and minimum unit is pixel, 
    else if unit is latlng, return unit is longtude and latitude, and minimum unit is longtude and latitude.

    returns : return pickup_tiles as np.array([[x1, y1], [x2, y2],...])
    """

pickup_tiles = lpi.overlappingTiles(bounds, zoom=zoom)
pickup_tiles

# %%
# return xy tile coordinate array and where intersection is in tile.
pickup_tiles, pickup_tiles_intersection = lpi.overlappingTileSegments(bounds, zoom=zoom)
print(pickup_tiles)
print(pickup_tiles_intersection)

# pickup_tiles_intersection as np.array([[x1, y1], [x2, y2],...]), 
# np.array([[x11,y11],[x12,y12],[x13,y13]], ...). x11, y11 and x12, y12 is intersections of bounds lines and tiles lines.
# x13, y13 is bounds corner points in the tiles. If intersections or corner points are not exist, it is in np.nan.
#%%
#def concat_tile(self, bounds, pickup_tiles, file_path, 
#                        file_extention='.webp', save_path='./',
#                        draw_polyline=False, draw_bounds=True, crop_mode='dst', zoom='self', TILE_SIZE='self'):
a=  """
    A function that collects tiles in bounds and connects the top of 
    the satellite image with the polyline.
    bounds (list) : returns for xy_aligned or terminal_node_aligned.
    pickup_tiles (list) : returns for overlappingTile function.
    
    file_path (str) : path for where there are satellite image tiles. ex. '../datasets/'
    file_extention (str) : Extension for tile image files. Probably, '.webp'.
    save_file_path (str) : path for save image file.
    
    draw_polyline (False or shapely.geometry.LineString) : polyline using make bounds.) If False, polyline is not drawn.
    draw_bounds (bool) : If True, bounds is drawn.
    crop_mode (list of str) 'croped','mask','dst','dst2' or False: If 'all' 4 all pattern images return.
    dst, black back crop, dst2, white back crop.
    if False, no croped.
    
    zoom (int)[0-18] : zoom level.
    TILE_SIZE (tuple of int) : The tile size(px). Probably, (256, 256).
    """



lpi.concat_tile(bounds, pickup_tiles, \
        file_path='../datasets/', file_extention='.webp', \
        save_path='./', draw_polyline=ls2, draw_bounds=True, crop_mode='dst')

#%%
lpi.concat_tile_segment(pickup_tiles, pickup_tiles_intersection, \
        file_path='../datasets/', file_extention='.webp', \
        save_path='./', draw_polyline=ls2, draw_bounds=True, crop_mode='dst')