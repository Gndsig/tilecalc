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
ls2 = LineString([(139.07455444335938, 35.9913409624497), (139.07730102539062, 35.99356320664446),(139.07867431640625,35.99722880246021),(139.0814208984375,35.99822880246021)])


#%%
# sample
print(ls2.area)
print(ls2.bounds)
print(ls2.length)
print(ls2.geom_type)
print(ls2.distance)
list(ls2.coords)


#%%
#def overlappingTileSegments
self = LinkingPolylineImage(ls2)
#bounds = self.terminal_node_aligned()
bounds = self.xy_aligned(form='rectangle')
zoom=18
filepath = './datasets'
file_extention = '.webp'
TILE_SIZE='self'


if TILE_SIZE=='self' and hasattr(self, 'TILE_SIZE'):
    TILE_SIZE = self.TILE_SIZE
elif TILE_SIZE=='self' and not hasattr(self, 'TILE_SIZE'):
    raise KeyError('TILE_SIZE is not found. Please input TILE_SIZE or in class instance')
        
if not filepath[-1:]=='/':
    filepath = filepath+'/'
filepath = filepath + str(zoom) + '/'

if len(bounds)==4:  # if bounds=(x_min,ymin,x_max,y_max)
    bounds = [0, np.array([[bounds[0],bounds[1]],[bounds[2],bounds[1]],[bounds[2],bounds[3]],[bounds[0], bounds[3]]]) ]

theta = bounds[0]
# warning :: pixel coordinate is y axis inversion.
pixel_bounds = np.array(list(map(functools.partial(self.latlng_to_pixel, zoom=zoom,is_round=False), bounds[1])))

# for make searching point, Make roughly tile points
bounds_bounds = self.xy_aligned(Polygon(pixel_bounds), form='minmax')
min_x_s = (bounds_bounds[0,0] - TILE_SIZE[0]) // TILE_SIZE[0] * TILE_SIZE[0]
min_y_s = (bounds_bounds[0,1] - TILE_SIZE[1]) // TILE_SIZE[1] * TILE_SIZE[1]
max_x_s = (bounds_bounds[1,0] + TILE_SIZE[0]) // TILE_SIZE[0] * TILE_SIZE[0]
max_y_s = (bounds_bounds[1,1] + TILE_SIZE[1]) // TILE_SIZE[1] * TILE_SIZE[1]

# searching points range on pixel coordinate
x_range = [ min_x_s, max_x_s, TILE_SIZE[0]]
y_range = [ min_y_s, max_y_s, TILE_SIZE[1]]

# Make coordinates of all searching points
x_set = np.arange(x_range[0], x_range[1]+0.001, x_range[2]).reshape(-1,1)
lx = x_set.shape[0]
y_set = np.arange(y_range[0], y_range[1]+0.001, y_range[2])
ly = y_set.shape[0]

# Create xy pair
points = np.empty((0,2),float)
for y in y_set:
    y_ = np.tile(y, lx).reshape(-1,1)
    xy = np.concatenate([x_set, y_], axis=1)
    points = np.concatenate([points,xy], axis=0)
len_points = lx*ly

# linking tile and points
# Tiles are represented by 4 points number
# 6---7---8 
# | 2'| 3'|
# 3---4---5
# | 0'| 1'|       
# 0---1---2
# 0,1,2 : points number, 0',1',2' : tiles number
# this case, tile 0' is [0, 1, 4, 3]

lx1 = lx-1
ly1 = ly-1
tile = []
tile_append = tile.append
for j in range(ly1):
    for i in range(lx1):
        tile_append([i +j+lx1*j, i+1 +j+lx1*j, i+2+lx1 +j+lx1*j, i+1+lx1 +j+lx1*j])
len_tile = len(tile)
tile = np.array(tile)

# judgement points are whether in bounds.
points_is_in_bounds = np.array(list(map(functools.partial(self.inpolygon, polygon=pixel_bounds), points)))

# tile have how many points in bounds.
howmany_points_in_bounds = []
howmany_points_in_bounds_append = howmany_points_in_bounds.append
for i in range(len_tile):
    howmuch_in_bounds = points_is_in_bounds[tile[i]].sum()
    howmany_points_in_bounds_append(howmuch_in_bounds)

# pick up tile having at least 1 points in bounds.
is_in_bounds = [hpib !=0 for hpib in howmany_points_in_bounds]

pickup_tile = points[tile[is_in_bounds,0]] / TILE_SIZE
pickup_tile = pickup_tile.astype(int)

pickup_tile_list = []
for pit in pickup_tile:
    x = pit[0]
    y = pit[1]
    path = filepath +str(x) + '/'+str(y) + file_extention
    pickup_tile_list.append(path)

if filepath:
    # all database tile.
    tilefile_list = glob.glob(filepath+'*/*')
    isnot_exist_files = set(pickup_tile_list) - set(tilefile_list)

    # if pickup_file is not exist, raise error
    if not isnot_exist_files==set():
        raise ValueError(str(isnot_exist_files)+' is not exist')
        
        
#%%




#%%
#path = Path(filepath)
#tilefile_list = list(path.glob('*/*'))
#tilefile_list_str = [str(x) for x in tilefile_list]
#tilefile_list_str
