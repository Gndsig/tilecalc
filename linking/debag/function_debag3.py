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

if not filepath:     
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

# linking tiles and points
# tiles are represented by 4 points number
# 6---7---8 
# | 2'| 3'|
# 3---4---5
# | 0'| 1'|       
# 0---1---2
# 0,1,2 : points number, 0',1',2' : tiles number
# this case, tiles 0' is [0, 1, 4, 3]


lx1 = lx-1
ly1 = ly-1
tiles = []
tiles_append = tiles.append
for j in range(ly1):
    for i in range(lx1):
        tiles_append([i +j+lx1*j, i+1 +j+lx1*j, i+2+lx1 +j+lx1*j, i+1+lx1 +j+lx1*j])
len_tiles = len(tiles)
tiles = np.array(tiles)

# linking tiles and lines, lines and points
lines = []  # all line segments composed of points
lines_append = lines.append
owner = []  # Which tile the lines belong to
owner_append = owner.append
for j in range(ly):
    for i in range(lx1):
        lines_append([i +j+lx1*j, i+1 +j+lx1*j])
        if j==0:
            owner_append([i +lx1*j])
        elif j==ly-1:
            owner_append([i-lx1 + lx1*j])
        else:
            owner_append([i-lx1 + lx1*j, i +lx1*j])
            
for j in range(ly1):
    for i in range(lx):
        lines_append([i +j+lx1*j, i+1+lx1 +j+lx1*j])
        if i==0:
            owner_append([i +lx1*j])
        elif i==lx-1:
            owner_append([i-1 + lx1*j])
        else:
            owner_append([i-1 + lx1*j, i +lx1*j])

len_lines = len(lines)
lines = np.array(lines)


# judgement points are whether in bounds.
points_is_in_bounds = np.array(list(map(functools.partial(self.inpolygon, polygon=pixel_bounds), points)))



# intersects with bounds when the end point of the line is True and False
line_points_in_bounds = []
line_points_in_bounds_append = line_points_in_bounds.append
for i in range(len_lines):
    howmany_in_bounds = points_is_in_bounds[lines[i]].sum()
    line_points_in_bounds_append(howmany_in_bounds)

# lines having cross==True points and False points, line is cross to bounds.
cross_lines = [lpib==1 for lpib in line_points_in_bounds]



# Judgment which bounds the intersecting line intersects
cross_lines_coords = points[lines[cross_lines]]
len_cross_lines_coords = len(cross_lines_coords)

# pixel_bounds having 4 lines, its represent points.
len_pixel_bounds = len(pixel_bounds)
pb_lines_coords = []
pb_lines_coords_append = pb_lines_coords.append
for i in range(len_pixel_bounds):
    if i!=len_pixel_bounds-1:
        pb_lines_coords_append([pixel_bounds[i],pixel_bounds[i+1]])
    elif i==len_pixel_bounds-1:
        pb_lines_coords_append([pixel_bounds[i],pixel_bounds[0]])

# judgment lines cross which bounds lines.
which_lines_cross = []
which_lines_cross_append = which_lines_cross.append
for j in range(len_cross_lines_coords):
    for i in range(len_pixel_bounds):
        is_lines_cross = self.is_intersected_ls(pb_lines_coords[i][0], pb_lines_coords[i][1], cross_lines_coords[j][0], cross_lines_coords[j][1])
        if is_lines_cross:
            which_lines_cross_append(i)
            break
        elif (not is_lines_cross) and i==len_pixel_bounds-1:
            which_lines_cross_append(np.nan)
        
        
# Find intersection
intersection = []
intersection_append = intersection.append
for j in range(len_cross_lines_coords):
    i = which_lines_cross[j]
    intersec = self.intersection_ls(pb_lines_coords[i][0], pb_lines_coords[i][1], cross_lines_coords[j][0], cross_lines_coords[j][1])
    intersection_append(intersec)
    
intersection = np.array(intersection)

# tile number having cross lines.
cross_owner = []
cross_owner_append = cross_owner.append
for own, is_cross in zip(owner, cross_lines):
    if is_cross:
        cross_owner_append(own)

# The coordinates of the intersection of tiles
tiles_intersection = [[] for i in range(len_tiles)]
for cross_line_num, cro_own in enumerate(cross_owner):
    for tile_num in cro_own:
        tiles_intersection[tile_num].append(intersection[cross_line_num])

# to calculate conveniently, convert numpy.array      
tiles_intersection = [ [[np.nan,np.nan],[np.nan,np.nan]] if ti==[] else ti for ti in tiles_intersection]
tiles_intersection = np.array(tiles_intersection)


# tile_base_point
tile_base_points = []
tile_base_points_append = tile_base_points.append
for tile_num in range(len_tiles):
    tile_base_points_append(points[tiles[tile_num][0]])
tile_base_points = np.array(tile_base_points)

# Convert to coordinates in tile
for tile_num in range(len_tiles):
    if np.isnan(tiles_intersection[tile_num]).all():
        continue
    tile_base_point = tile_base_points[tile_num]
    
    for i in range(len(tile_base_point)):  # 2d:i=0,1
        tiles_intersection[tile_num][:,i] -= tile_base_point[i]


# Calculate which position of which tile is relative to the 4 points of Bounds.
bounds_corner_tiles_index = []
bounds_corner_tile_coords = []
for i in range(len_pixel_bounds):  # i=0,1,2,3 if rectangle
    corner_jugdment = -(tile_base_points - pixel_bounds[i])
    tile_corner_index = np.where(np.all((0.0 <= corner_jugdment) * (corner_jugdment[:,[0]] < TILE_SIZE[0]) * (corner_jugdment[:,[1]] < TILE_SIZE[1]), axis=1) )
    bounds_corner_tiles_index.append(tile_corner_index)
    bounds_corner_tile_coords.append(corner_jugdment[tile_corner_index])
    
bounds_corner_tiles_index = np.array(bounds_corner_tiles_index).reshape(-1,)
bounds_corner_tile_coords = np.array(bounds_corner_tile_coords).reshape(-1,2)

# Add nan to the coordinates of the third point for the time being
tiles_intersection = np.concatenate([tiles_intersection, np.full( (len_tiles,1,2) ,np.nan)], axis=1)
#%%
bounds_corner_tile_coords
#%%
for i in range(len(bounds_corner_tile_coords)):
    bounds_tile_num = bounds_corner_tiles_index[i]
    tiles_intersection[bounds_tile_num][2] = bounds_corner_tile_coords[i]
#%%
tiles_intersection

#%%
# check
#for i in range(len_pixel_bounds):  # i=0,1,2,3 if rectangle
#    if not self.inpolygon(pixel_bounds[i] , points[tiles[bounds_corner_tiles[i]]]):
#        raise ValueError

#%%

# tiles have how many points in bounds.
howmany_points_in_bounds = []
howmany_points_in_bounds_append = howmany_points_in_bounds.append
for i in range(len_tiles):
    howmuch_in_bounds = points_is_in_bounds[tiles[i]].sum()
    howmany_points_in_bounds_append(howmuch_in_bounds)

# pick up tiles having at least 1 points in bounds.
is_in_bounds = [hpib !=0 for hpib in howmany_points_in_bounds]

# Tile having bounds corner contains, because the case is exists that 
# tile all points is not in bounds, but bounds corner points is in the tile.
for index in bounds_corner_tiles_index:
    is_in_bounds[index] = True
    

pickup_tiles = points[tiles[is_in_bounds,0]] / TILE_SIZE
pickup_tiles = pickup_tiles.astype(int)

pickup_tiles_intersection = tiles_intersection[is_in_bounds]


#return pickup_tiles, pickup_tiles_intersection










COMMENTOUT="""
pickup_tiles_list = []
for pit in pickup_tiles:
    x = pit[0]
    y = pit[1]
    path = filepath +str(x) + '/'+str(y) + file_extention
    pickup_tiles_list.append(path)


# all database tiles.
tilesfile_list = glob.glob(filepath+'*/*')
isnot_exist_files = set(pickup_tiles_list) - set(tilesfile_list)

# if pickup_file is not exist, raise error
if not isnot_exist_files==set():
    raise ValueError(str(isnot_exist_files)+' is not exist')"""
        



#%%
#path = Path(filepath)
#tilesfile_list = list(path.glob('*/*'))
#tilesfile_list_str = [str(x) for x in tilesfile_list]
#tilesfile_list_str

#%%
#lines_intersection = np.array([[np.nan, np.nan] for i in range(len_lines)])
#lines_intersection[cross_lines] = intersection

#%%
COMMENT_OUT="""
tiles_owner = [[] for i in range(len_tiles)]  # Which line the tile has
for i in range(len_lines):
    len_owner_i = len(owner[i])
    if len_owner_i==1:
        tiles_owner[owner[i][0]].append(i)  # [0] is peel list
    elif len_owner_i==2:
        tiles_owner[owner[i][0]].append(i)
        tiles_owner[owner[i][1]].append(i)
    else:
        raise KeyError('owner is max 2 length')"""
        
        