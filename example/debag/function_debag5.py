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
bounds_ = lpi.xy_aligned(ls1)
bounds_

#%%
#xy_aligned
self=LinkingPolylineImage(ls1)
polyline=ls1
form='rectangle'
minimum=[]
#minimum = [10,10]
unit=['latlng', 'pixel']
zoom=18

a=  """function to Minimum Bounding Rectangle aligned xy axis.
    polyline (shapely.geometry.LineString) : LineString object. If default, use class instance.
    
    form (str)['minmax', 'rectangle'] : If 'minmax', return np.array([[x_min, y_min], [x_max, y_max]]).
    If 'rectangle', return [theta(=0), np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]]) ]
    Theta is angle[rad] from x axis, this case is 0.
    
    zoom and minimum_unit is using only mimimum!=[]
    
    minimum (list of [x_min,y_min]) : Minimum width of Minimum Bounding Rectangle.
    unit (list of str) [['latlng', 'pixel'], ['pixel', 'pixel'] ] : The first in the list shows the unit of input.
    THe second in the list shows the unit of output.
    And minimum unit follows the second in the list.
    If unit is ['latlng','pixel'], the inputed polyline unit is longtude and latitude, and return unit is pixel and minimum unit is pixel, 
    else if unit is latlng, return unit is longtude and latitude, and minimum unit is longtude and latitude.
    
    Finaly, the unit of minimum and return should be unified.
    Warning, If unit is "latlng", later convert "latlng" to "pixel", point of rectangle is a little shift, no more parallelograms.
    
    If width < minimum, width=minimum and Rectangle is helf width from center point.
    """

#if polyline=='self' and hasattr(self, 'polyline'):
#    polyline = self.polyline
#elif polyline=='self' and not hasattr(self, 'polyline'):
#    raise KeyError('polyline is not found. Please input polyline or in class instance')

if isinstance(polyline, str):
    polyline = shapely.wkt.loads(polyline)


raw_bounds = polyline.bounds  # (x_min, y_min, xmax, y_max)
raw_bounds = (raw_bounds[0], raw_bounds[1]), (raw_bounds[2],raw_bounds[3])

raw_bounds = self.unit_change(raw_bounds, unit=unit,zoom=zoom, is_round=False)

#if unit[0]==unit[1]:
#    pass
#elif unit[0]=='latlng' and unit[1]=='pixel':
#    raw_bounds = tuple(list(map(functools.partial(self.latlng_to_pixel, zoom=zoom,is_round=False), raw_bounds)))
#elif unit[0]=='pixel' and unit[1]=='latlng':
#    raw_bounds = tuple(list(map(functools.partial(self.pixel_to_latlng, zoom=zoom), raw_bounds)))
#else:
#    raise ValueError('unit is "pixel" or "latlng" of 2 pieces list ex, ["latlng","pixel"]')

x_min = raw_bounds[0][0]
y_min = raw_bounds[0][1]
x_max = raw_bounds[1][0]
y_max = raw_bounds[1][1]

center_point = ( (x_min+x_max)/2.0 , (y_min+y_max)/2.0 )

width = [ x_max-x_min, y_max-y_min ]

if not(minimum==() or minimum==[] or minimum==None):
    try:
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


#%%
Polygon(bounds[1])

#%%

#@staticmethod
#def rotation_axis_matrix_2d(theta):
#    return np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])


# terminal_node_aligned

self=LinkingPolylineImage(ls1)
polyline = ls1
minimum=[]
unit=['latlng','pixel']
zoom=18

a="""function to Minimum Bounding Rectangle aligned vector start to end.
    polyline (shapely.geometry.LineString) : LineString object. If default, use class instance.
    minimum (list of [x_min,y_min]) : Minimum width of Minimum Bounding Rectangle.
    If width < minimum, width=minimum and Rectangle is helf width from center point.
    
    unit (str) ['latlng', 'pixel'] : specify minimum unit.'latlng' is latitude and longitude. 
    If unit is pixel, return unit is pixel and minimum unit is pixel, 
    else if unit is latlng, return unit is longtude and latitude, and minimum unit is longtude and latitude.
    
    Finaly, the unit of minimum and return should be unified.
    Warning, If unit is "latlng", later convert "latlng" to "pixel", point of rectangle is a little shift, no more parallelograms.
    
    returns : [theta, np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]]) ]
    Theta is angle[rad] vector start to end from x axis.
    """

#if polyline=='self' and hasattr(self, 'polyline'):
#    polyline = self.polyline
#elif polyline=='self' and not hasattr(self, 'polyline'):
#    raise KeyError('polyline is not found. Please input polyline or in class instance')

if isinstance(polyline, str):
    polyline = shapely.wkt.loads(polyline)
coords = np.array(polyline.coords)  # all LineString coordinate.

coords = np.array(self.unit_change(coords, unit=unit,zoom=zoom, is_round=False))

start_coord = coords[0]
end_coord = coords[-1]
# vector start to end
vec_se = np.array([ end_coord[0]-start_coord[0], end_coord[1]-start_coord[1] ])
# theta from x axis [rad]
theta = np.arctan2( vec_se[1], vec_se[0] )

rotation_axis_matrix = self.rotation_axis_matrix_2d(theta)

# Coordinate transformation using rotation.
coords_trans = np.dot(rotation_axis_matrix, coords.T).T


# get_bounds
bounds_trans = self.xy_aligned( LineString(coords_trans),form='rectangle', \
        minimum=minimum, unit=['same','same'], zoom=zoom)
# reverse coordinate transformation
rotation_axis_matrix_rev = self.rotation_axis_matrix_2d(-theta)
bounds_ = np.dot(rotation_axis_matrix_rev, bounds_trans[1].T).T
bounds = [theta, bounds_]

#%%
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

# check_latlng
self=LinkingPolylineImage()
coordinate = bounds3

a="""
coordinate (tuple or list of np.array of float) : coordinate. xy order. For example, (140.087099, 36.104665) 

return : if coordinate is longtude and latitude, return True. Else if, coordinate is pixle return False 
"""

np_shape = np.array(coordinate).shape
np_shape_rear = np.array(coordinate[1]).shape


def check_unit(coordinate):
    # check coordinate is longtude and latitude
    if all([-180 <= coord and coord <= 180 for coord in coordinate]):
        return True
    else:
        return False

if len(np_shape)==2:
    # if coordinate is multiple coordinates ex. [[2,2],[2,2]]

    is_latlngs = list(map(check_unit, coordinate))
    is_latlng = all(is_latlngs)

else:
    if len(np_shape_rear)==0:
        # if coordinate is single coordinate ex. [1,1]
        is_latlng = check_unit(coordinate)
        
    elif len(np_shape_rear)==2:
        # if coordinate is bounds ex. [0, np.array([[1,1],[2,2]]) ]
        print('This instance is bounds ex. [ theta, np.array([[1,1],[2,2]])]')
        is_latlngs = list(map(check_unit, coordinate[1]))
        is_latlng = all(is_latlngs)
    else:
        # other
        raise ValueError('Input must be coordinate ex. [1,1] or [[2,2],[2,2]] [0, np.array([[2,2],[2,2]])],')

is_latlng


#%%
# return xy tile coordinate array.
#pickup_tiles_ = lpi.overlappingTiles(bounds1, zoom=zoom)
#pickup_tiles_


#%%
bounds3 = lpi.xy_aligned(polyline=ls1,unit=['latlng','latlng'], form='minmax')

#%%

# overLappingTiles
self=LinkingPolylineImage(ls1)
bounds=bounds1
unit = ['pixel', 'pixel']
zoom='self'
TILE_SIZE='self'
a="""
bounds (list) : input [theta, np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]])] by xy_aligned or terminal node aligned.
zoom (int)[0-18] : zoom level.
filepath (str of False) : tiles database path. If path is --/--/--/zoom/x/y, the part is --/--/--/.
Don't need it, if not False, check whether file exists.
file_extention (str) : If use filepath, specify file extention.

returns : If filepath=False, return pickup_tiles as np.array([[x1, y1], [x2, y2],...])
elif filepash='path', return filepath list as ['filepath/zoom/x1/y1',...]
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

#%%
pixel_bounds = np.array(list(map(functools.partial(self.latlng_to_pixel, zoom=zoom,is_round=False), bounds[1])))


#%%
points, len_points, tiles, len_tiles = self._make_tile_mesh(pixel_bounds, TILE_SIZE, is_lines=False)


# judgement points are whether in bounds.
points_is_in_bounds = np.array(list(map(functools.partial(self.inpolygon, polygon=pixel_bounds), points)))

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
len_pixel_bounds = len(pixel_bounds)

bounds_corner_tiles_index = []
for i in range(len_pixel_bounds):  # i=0,1,2,3 if rectangle
    corner_jugdment = -(tile_base_points - pixel_bounds[i])
    tile_corner_index = np.where(np.all((0.0 <= corner_jugdment) * (corner_jugdment[:,[0]] < TILE_SIZE[0]) * (corner_jugdment[:,[1]] < TILE_SIZE[1]), axis=1) )
    bounds_corner_tiles_index.append(tile_corner_index)
    
bounds_corner_tiles_index = np.array(bounds_corner_tiles_index).reshape(-1,)

for index in bounds_corner_tiles_index:
    is_in_bounds[index] = True  
# -------------------------------------------------------

pickup_tiles = points[tiles[is_in_bounds,0]] / TILE_SIZE
pickup_tiles = pickup_tiles.astype(int)

return pickup_tiles




#%%



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