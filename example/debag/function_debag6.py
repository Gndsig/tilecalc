# using masic command #%% as vscode like jupyter.

# 長方形をpixelの方で作るためのデバッグ

#%%
import os
import sys
#os.getcwd()
os.chdir("/workdir/linking_polyline_image/linking/")

from link import *


ls1 = 'LINESTRING(139.07650102539062 35.99722956359412, 139.0784208984375 35.995022880246021, 139.0814208984375 35.994022880246021)'
ls2 = LineString([(139.07585444335938, 35.9923409624497), (139.07730102539062, 35.99356320664446),(139.07867431640625,35.99722880246021),(139.0824208984375,35.99842880246021)])



#%%
lpi = LinkingPolylineImage(ls1)
bounds = lpi.xy_aligned(ls1)
bounds

#%%
Polygon(bounds[1])

#%%
lpi = LinkingPolylineImage(ls1)
zoom=18

bounds1 = lpi.xy_aligned(polyline=ls1, form='rectangle', minimum=[], unit=['latlng','pixel'], zoom=zoom)  # if not specify, class instance use.
bounds1_ = lpi.xy_aligned(polyline=ls1, form='minmax')
bounds2 = lpi.terminal_node_aligned(ls2, unit=['latlng','pixel'])
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
# return xy tile coordinate array.
#pickup_tiles_ = lpi.overlappingTiles(bounds1, zoom=zoom)
#pickup_tiles_


#%%
bounds3 = lpi.xy_aligned(polyline=ls1,unit=['latlng','latlng'], form='minmax')

#%%
bounds3

#%%

# overlappingTiles
self=LinkingPolylineImage(ls1)
bounds=bounds1
unit = ['pixel','pixel']
zoom='self'
TILE_SIZE='self'
a="""
bounds (list) : input [theta, np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]])] by xy_aligned or terminal node aligned.
zoom (int)[0-18] : zoom level.
filepath (str of False) : tiles database path. If path is --/--/--/zoom/x/y, the part is --/--/--/.
Don't need it, if not False, check whether file exists.
file_extention (str) : If use filepath, specify file extention.

returns : return pickup_tiles as np.array([[x1, y1], [x2, y2],...])
"""

if TILE_SIZE=='self' and hasattr(self, 'TILE_SIZE'):
    TILE_SIZE = self.TILE_SIZE
elif TILE_SIZE=='self' and not hasattr(self, 'TILE_SIZE'):
    raise KeyError('TILE_SIZE is not found. Please input TILE_SIZE or in class instance')

if zoom=='self' and hasattr(self, 'zoom'):
    zoom = self.zoom
elif zoom=='self' and not hasattr(self, 'zoom'):
    raise KeyError('zoom is not found. Please input zoom or in class instance')


if len(bounds)==4:  # if bounds=(x_min,ymin,x_max,y_max)
    bounds = [0, np.array([[bounds[0],bounds[1]],[bounds[2],bounds[1]],[bounds[2],bounds[3]],[bounds[0], bounds[3]]]) ]

theta = bounds[0]

# probably, bounds is pixel, from here bounds must be pixel unit.
bounds_pixel = np.array(self.unit_change(bounds[1], unit=unit,zoom=zoom, is_round=False))

points, len_points, tiles, len_tiles = self._make_tile_mesh(bounds_pixel, TILE_SIZE, is_lines=False)

# judgement points are whether in bounds.
points_is_in_bounds = np.array(list(map(functools.partial(self.inpolygon, polygon=bounds_pixel), points)))

# tiles have how many points in bounds.
howmany_points_in_bounds = []
howmany_points_in_bounds_append = howmany_points_in_bounds.append
for i in range(len_tiles):
    howmuch_in_bounds = points_is_in_bounds[tiles[i]].sum()
    howmany_points_in_bounds_append(howmuch_in_bounds)

# pick up tiles having at least 1 points in bounds.
is_in_bounds = [hpib !=0 for hpib in howmany_points_in_bounds]


# ---------Tile having bounds corner contains, because the case is exists that 
# tile all points is not in bounds, but bounds corner points is in the tile.--------

# tile_base_point
tile_base_points = []
tile_base_points_append = tile_base_points.append
for tile_num in range(len_tiles):
    tile_base_points_append(points[tiles[tile_num][0]])
tile_base_points = np.array(tile_base_points)

# Calculate which position of which tile is relative to the 4 points of Bounds.
len_bounds_pixel = len(bounds_pixel)

bounds_corner_tiles_index = []
for i in range(len_bounds_pixel):  # i=0,1,2,3 if rectangle
    corner_jugdment = -(tile_base_points - bounds_pixel[i])
    tile_corner_index = np.where(np.all((0.0 <= corner_jugdment) * (corner_jugdment[:,[0]] < TILE_SIZE[0]) * (corner_jugdment[:,[1]] < TILE_SIZE[1]), axis=1) )
    bounds_corner_tiles_index.append(tile_corner_index)
    
bounds_corner_tiles_index = np.array(bounds_corner_tiles_index).reshape(-1,)

for index in bounds_corner_tiles_index:
    is_in_bounds[index] = True  
# -------------------------------------------------------

pickup_tiles = points[tiles[is_in_bounds,0]] / TILE_SIZE
pickup_tiles = pickup_tiles.astype(int)

pickup_tiles



#%%

#overlappingTileSegments
self=LinkingPolylineImage(ls1)
bounds=bounds1
unit = ['pixel','pixel']
zoom='self'
TILE_SIZE='self'

a="""
bounds (list) : input [theta, np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]])] by xy_aligned or terminal node aligned.
zoom (int)[0-18] : zoom level.

returns : If filepath=False, return pickup_tiles, pickup_tiles_intersection as np.array([[x1, y1], [x2, y2],...]), 
np.array([[x11,y11],[x12,y12],[x13,y13]], ...). x11, y11 and x12, y12 is intersections of bounds lines and tiles lines.
x13, y13 is bounds corner points in the tiles. If intersections or corner points are not exist, it is in np.nan.
elif filepath='anypath', return pickup_tiles, pickup_tiles_intersection, pickup_tiles_list. pickup_tile_list as ['filepath/zoom/x1/y1',...].
"""


if TILE_SIZE=='self' and hasattr(self, 'TILE_SIZE'):
    TILE_SIZE = self.TILE_SIZE
elif TILE_SIZE=='self' and not hasattr(self, 'TILE_SIZE'):
    raise KeyError('TILE_SIZE is not found. Please input TILE_SIZE or in class instance')

if zoom=='self' and hasattr(self, 'zoom'):
    zoom = self.zoom
elif zoom=='self' and not hasattr(self, 'zoom'):
    raise KeyError('zoom is not found. Please input zoom or in class instance')

if len(bounds)==4:  # if bounds=(x_min,ymin,x_max,y_max)
    bounds = [0, np.array([[bounds[0],bounds[1]],[bounds[2],bounds[1]],[bounds[2],bounds[3]],[bounds[0], bounds[3]]]) ]

theta = bounds[0]

# warning :: pixel coordinate is y axis inversion.
# probably, bounds is pixel, from here bounds must be pixel unit.
bounds_pixel = np.array(self.unit_change(bounds[1], unit=unit,zoom=zoom, is_round=False))

points, len_points, tiles, len_tiles, lines, len_lines, owner = self._make_tile_mesh(bounds_pixel, TILE_SIZE, is_lines=True)


# judgement points are whether in bounds.
points_is_in_bounds = np.array(list(map(functools.partial(self.inpolygon, polygon=bounds_pixel), points)))

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
len_pixel_bounds = len(bounds_pixel)
pb_lines_coords = []
pb_lines_coords_append = pb_lines_coords.append
for i in range(len_pixel_bounds):
    if i!=len_pixel_bounds-1:
        pb_lines_coords_append([bounds_pixel[i],bounds_pixel[i+1]])
    elif i==len_pixel_bounds-1:
        pb_lines_coords_append([bounds_pixel[i],bounds_pixel[0]])

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
    corner_jugdment = -(tile_base_points - bounds_pixel[i])
    tile_corner_index = np.where(np.all((0.0 <= corner_jugdment) * (corner_jugdment[:,[0]] < TILE_SIZE[0]) * (corner_jugdment[:,[1]] < TILE_SIZE[1]), axis=1) )
    bounds_corner_tiles_index.append(tile_corner_index)
    bounds_corner_tile_coords.append(corner_jugdment[tile_corner_index])
    
bounds_corner_tiles_index = np.array(bounds_corner_tiles_index).reshape(-1,)
bounds_corner_tile_coords = np.array(bounds_corner_tile_coords).reshape(-1,2)

# Add nan to the coordinates of the third point for the time being
tiles_intersection = np.concatenate([tiles_intersection, np.full( (len_tiles,1,2) ,np.nan)], axis=1)

for i in range(len(bounds_corner_tile_coords)):
    bounds_tile_num = bounds_corner_tiles_index[i]
    tiles_intersection[bounds_tile_num][2] = bounds_corner_tile_coords[i]

# check
#for i in range(len_pixel_bounds):  # i=0,1,2,3 if rectangle
#    if not self.inpolygon(pixel_bounds[i] , points[tiles[bounds_corner_tiles[i]]]):
#        raise ValueError

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

pickup_tiles, pickup_tiles_intersection



# %%
# return xy tile coordinate array and where intersection is in tile.
pickup_tiles, pickup_tiles_intersection = lpi.overlappingTileSegments(bounds1, zoom=zoom)
print(pickup_tiles)
print(pickup_tiles_intersection)

# pickup_tiles_intersection as np.array([[x1, y1], [x2, y2],...]), 
# np.array([[x11,y11],[x12,y12],[x13,y13]], ...). x11, y11 and x12, y12 is intersections of bounds lines and tiles lines.
# x13, y13 is bounds corner points in the tiles. If intersections or corner points are not exist, it is in np.nan.
# elif filepath='anypath', return pickup_tiles, pickup_tiles_intersection, pickup_tiles_list. pickup_tile_list as ['filepath/zoom/x1/y1',...].

#%%
# A series of processing
lpi2 = LinkingPolylineImage(ls2, zoom=zoom)
bounds = lpi2.terminal_node_aligned()
#bounds = lpi2.xy_aligned()
pickup_tiles, pickup_tiles_intersection = lpi2.overlappingTileSegments(bounds)


# %%

filepath='../datasets/'
file_extention='.webp'
zoom=zoom
pickup_tiles
TILE_SIZE=(256, 256)
self = LinkingPolylineImage(ls2, zoom=zoom)

#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw

def concat_h_blank(im1, im2, color=(0,0,0)):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def concat_v_blank(im1, im2, color=(0,0,0)):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def concat_h_multi_blank(im_list, color=(0,0,0)):
    _im = im_list.pop(0)
    for im in im_list:
        _im = concat_h_blank(_im, im, color)
    return _im

def concat_v_multi_blank(im_list, color=(0,0,0)):
    _im = im_list.pop(0)
    for im in im_list:
        _im = concat_v_blank(_im, im, color)
    return _im

def concat_tile_blank(im_list_2d, color=(0,0,0)):
    im_list_v = [concat_h_multi_blank(im_list_h, color) for im_list_h in im_list_2d]
    return concat_v_multi_blank(im_list_v, color)

def make_dummy(im, color=(0,0,0)):
    dst = Image.new('RGB', (im.width, im.height), color)
    return dst



def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

pickup_tiles_list = self._pickup_file_search(pickup_tiles, zoom, filepath, file_extention)

img_list = [Image.open(pickup_tile) for pickup_tile in pickup_tiles_list]
dummy = make_dummy(img_list[0])


min_x = pickup_tiles[:,0].min()
min_y = pickup_tiles[:,1].min()
max_x = pickup_tiles[:,0].max()
max_y = pickup_tiles[:,1].max()

width_x = max_x - min_x
width_y = max_y - min_y

pt_standard = pickup_tiles.copy()
pt_standard[:,0] -= min_x
pt_standard[:,1] -= min_y

arrangement_image = [[dummy for i in range(width_x+1)] for j in range(width_y+1)]

len_pickup_tiles = len(pickup_tiles)
for i in range(len_pickup_tiles):
    arrangement_image[pt_standard[i,1]][pt_standard[i,0]] = img_list[i]

concated_image = concat_tile_blank(arrangement_image)

#%%
# draw polyline
polyline = ls2
polyline_coords = np.array(polyline.coords)
pixel_coords = np.array(list(map(functools.partial(self.latlng_to_pixel, zoom=zoom,is_round=False), polyline_coords)))

pixel_coords[:,0] -= min_x * TILE_SIZE[0]
pixel_coords[:,1] -= min_y * TILE_SIZE[1]
pixel_coords_tuple = tuple(map(tuple, pixel_coords))

polyline_image = concated_image.copy()
draw = ImageDraw.Draw(polyline_image)
draw.line(pixel_coords_tuple, fill=(255, 255, 0), width=10)
polyline_image

# drow bounds
theta = bounds[0]
pixel_bounds = np.array(list(map(functools.partial(self.latlng_to_pixel, zoom=zoom,is_round=False), bounds[1])))

pixel_bounds[:,0] -= min_x * TILE_SIZE[0]
pixel_bounds[:,1] -= min_y * TILE_SIZE[1]
pixel_bounds_drow = np.concatenate([pixel_bounds, pixel_bounds[[0]]], axis=0)
pixel_bounds_tuple = tuple(map(tuple, pixel_bounds_drow))


draw = ImageDraw.Draw(polyline_image)
draw.line(pixel_bounds_tuple, fill=(255, 0, 255), width=10)
polyline_image

#%%
image_cv = pil2cv(polyline_image)
#image_cv = pil2cv(concated_image)
pixel_bounds = pixel_bounds.astype(int)

def polygon_crop_cv(img, polygon, mode='all'):
    """
    mode (list of str) 'croped','mask','dst','dst2' : If 'all' 4 all pattern images return.
    dst : black back crop, dst2 : white back crop.
    """
    if mode=='all':
        mode=['croped', 'mask', 'dst', 'dst2']
    elif mode!='all' and isinstance(mode,str):
        mode = [mode]
    
    returns = []
    rect = cv2.boundingRect(polygon)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    if 'croped' in mode:
        ## (1) Crop the bounding rect
        returns.append(croped)
    
    polygon = polygon - polygon.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)  
    if 'mask' in mode:
        ## (2) make mask
        cv2.drawContours(mask, [polygon], -1, (255, 255, 255), -1, cv2.LINE_AA)
        returns.append(mask)
    
    if 'dst' in mode:
        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        returns.append(dst)
        
    if 'dst2' in mode:
        ## (4) add the white background
        bg = np.ones_like(croped, np.uint8)*255
        cv2.bitwise_not(bg,bg, mask=mask)
        dst2 = bg+ dst
        returns.append(dst2)
    return returns

croped, mask, dst, dst2 = polygon_crop_cv(image_cv, pixel_bounds)
#%%
savepath = './images/'
cv2.imwrite(savepath+"croped.png", croped)
cv2.imwrite(savepath+"mask.png", mask)

cv2.imwrite(savepath+"dst.png", dst)
cv2.imwrite(savepath+"dst2.png", dst2)

# バッファ






#%%
FOR_DEBAG="""
arrangement_image = [[0 for i in range(width_x+1)] for j in range(width_y+1)]

len_pickup_tiles = len(pickup_tiles)
for i in range(len_pickup_tiles):
    arrangement_image[pt_standard[i,1]][pt_standard[i,0]] = str(pt_standard[i,0])+','+str(pt_standard[i,1])"""
#%%
l = img_list
concat_tile_blank([ [l[2], l[3], l[4]]])

concat_tile_blank([ [l[0], l[1]],
                    [l[3],l[4]], 
                    [l[7],l[8]]])




#%%
def concat_h_blank(im1, im2, color=(0,0,0)):
    im1_height,im1_width = im1.shape[:2]
    im2_height,im2_width = im2.shape[:2]
    height = max(im1_height, im2_height)
    width = im1_width + im2_width
    color = np.array(color)
    new_img = np.tile(color, (height,width,1))
    new_img[0:im1_height, 0:im1_width] = im1
    new_img[0:im2_height, im1_width:] = im2
    return new_img

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


def make_dummy(im, color=(0,0,0)):
    height, width = im.shape[:2]
    color = np.array(color)
    return np.tile(color, (height,width,1))















#%%


#%%


#%%


#%%
# old function
    def ___xy_aligned(self, polyline='self',form='rectangle', minimum=[], minimum_unit='latlng', zoom=18):
        """function to Minimum Bounding Rectangle aligned xy axis.
        polyline (shapely.geometry.LineString) : LineString object. If default, use class instance.
        
        form (str)['minmax', 'rectangle'] : If 'minmax', return np.array([[x_min, y_min], [x_max, y_max]]).
        If 'rectangle', return [theta(=0), np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]]) ]
        Theta is angle[rad] from x axis, this case is 0.
        
        zoom and minimum_unit is using only mimimum!=[]
        
        minimum (list of [x_min,y_min]) : Minimum width of Minimum Bounding Rectangle.
        minimum_unit (str) ['latlng', 'pixel', 'tile'] : specify minimum unit.'latlng' is latitude and longitude. 
        If width < minimum, width=minimum and Rectangle is helf width from center point.
        """
        if polyline=='self' and hasattr(self, 'polyline'):
            polyline = self.polyline
        elif polyline=='self' and not hasattr(self, 'polyline'):
            raise KeyError('polyline is not found. Please input polyline or in class instance')
        
        if isinstance(polyline, str):
            polyline = shapely.wkt.loads(polyline)

        bounds = polyline.bounds
        x_min = bounds[0]
        y_min = bounds[1]
        x_max = bounds[2]
        y_max = bounds[3]

        center_point = ( (x_min+x_max)/2.0 , (y_min+y_max)/2.0 )

        width = [ x_max-x_min, y_max-y_min]

        if not(minimum==() or minimum==[] or minimum==None):
            try:
                if minimum_unit=='pixel':
                    minimum = self.pixel_to_latlng(minimum, zoom=zoom)
                elif minimum_unit=='tile':
                    minimum = self.tile_to_latlng(minimum, zoom=zoom)
            
            # correct mimimum bounds
                if width[0] < minimum[0]:
                    width[0] = minimum[0]
                if width[1] < minimum[1]:
                    width[1] = minimum[1]
            except:
                raise KeyError('minimum is (x_min, y_min) of tuple or list')
            
            x_min = center_point[0]-width[0]/2.0
            y_min = center_point[1]-width[1]/2.0
            x_max = center_point[0]+width[0]/2.0
            y_max = center_point[1]+width[1]/2.0
        bounds = np.array([[x_min, y_min], [x_max, y_max]])

        if form=='rectangle':
            # point order is counterclockwise.
            # [theta from x axis, np.array([4 points of spuare])]
            bounds = [0, np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]]) ]
        
        return bounds