# using masic command #%% as vscode like jupyter.
#%%
import os
import sys
#os.getcwd()
os.chdir("/workdir/linking_polyline_image/linking/")

from link_old import *

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
# return xy tile coordinate array.
pickup_tiles_ = lpi.overlappingTiles(bounds1, zoom=zoom)
pickup_tiles_

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
#bounds = lpi2.terminal_node_aligned()
bounds = lpi2.xy_aligned()
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

def polygon_crop_cv(img, polygon, mode='dst'):
    """
    mode (list of str) 'croped','mask','dst','dst2' : If 'all' 4 all pattern images return.
    dst : black back crop, dst2 : white back crop.
    """
    if mode=='all':
        mode=['croped', 'mask', 'dst', 'dst2']
    elif mode!='all' and isinstance(mode,str):
        mode = [mode]
    
    returns = []
    if 'croped' in mode:
        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(polygon)
        x,y,w,h = rect
        croped = img[y:y+h, x:x+w].copy()
        returns.append(croped)
        
    if 'mask' in mode:
        ## (2) make mask
        polygon = polygon - polygon.min(axis=0)
        mask = np.zeros(croped.shape[:2], np.uint8)
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