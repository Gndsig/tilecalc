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

#bounds = bounds1.copy()
bounds = bounds2.copy()

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


#%%
self=LinkingPolylineImage()
bounds
pickup_tiles
file_path='../datasets'
file_extention='.webp'
save_path='./'
draw_polyline=polyline
draw_bounds=True
crop_mode='all'
zoom='self'
TILE_SIZE='self'


rotate='horizontal'
return_check=False

"""
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

rotate (str or float or False)['theta','nearest', 'vertical', 'horizontal'] : If float, cropped image rotate input angle[rad].
If theta, rotate Angle between start point and end point of polyline,  if nearest, rotate along nearest x or y axis,
if vertical, rotate along the nearest x axis, as well, horizontal y axis.

### Warning ; Pixel coordinate is Y-axis flip. 
The image you see has the coordinates without y-axis flipping, 
but the numbers themselves are in the y-axis flipping coordinate system.
In short, the angle is in the reverse rotation direction as usual.

return_check (bool) : If True, return Pillow image to check.

zoom (int)[0-18] : zoom level.
TILE_SIZE (tuple of int) : The tile size(px). Probably, (256, 256).
"""
if zoom=='self' and hasattr(self, 'zoom'):
    zoom = self.zoom
elif zoom=='self' and not hasattr(self, 'zoom'):
    raise KeyError('zoom is not found. Please input zoom or in class instance')

if TILE_SIZE=='self' and hasattr(self, 'TILE_SIZE'):
    TILE_SIZE = self.TILE_SIZE
elif TILE_SIZE=='self' and not hasattr(self, 'TILE_SIZE'):
    raise KeyError('TILE_SIZE is not found. Please input TILE_SIZE or in class instance')

# the last of file_path '/' may or may not be.
pickup_tiles_list = self._pickup_file_search(pickup_tiles, zoom, file_path, file_extention)

# concat image
img_list = [Image.open(pickup_tile) for pickup_tile in pickup_tiles_list]
dummy = self._make_dummy(img_list[0])

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

concated_image = self._concat_tile_blank(arrangement_image)

polyline_image = concated_image.copy()

# draw polyline
if type(draw_polyline)==str:
    draw_polyline = shapely.wkt.loads(draw_polyline)
    
if type(draw_polyline)==shapely.geometry.linestring.LineString:
    polyline = draw_polyline
    polyline_coords = np.array(polyline.coords)
    pixel_coords = np.array(list(map(functools.partial(self.latlng_to_pixel, zoom=zoom,is_round=False), polyline_coords)))

    # change tile coordinate to pixel coordinate, from min point is (0,0).
    pixel_coords[:,0] -= min_x * TILE_SIZE[0]
    pixel_coords[:,1] -= min_y * TILE_SIZE[1]
    pixel_coords_tuple = tuple(map(tuple, pixel_coords))

    draw = ImageDraw.Draw(polyline_image)
    draw.line(pixel_coords_tuple, fill=(255, 255, 0), width=10)
elif draw_polyline==True:
    raise ValueError('draw_polyline is False or shapely.geometry.linestring.LineString')


# draw bounds
bounds_pixel = bounds[1].copy()
# change tile coordinate to pixel coordinate, from min point is (0,0).
bounds_pixel[:,0] -= min_x * TILE_SIZE[0]
bounds_pixel[:,1] -= min_y * TILE_SIZE[1]

if draw_bounds:
    bounds_pixel_drow = np.concatenate([bounds_pixel, bounds_pixel[[0]]], axis=0)
    bounds_pixel_tuple = tuple(map(tuple, bounds_pixel_drow))

    draw = ImageDraw.Draw(polyline_image)
    draw.line(bounds_pixel_tuple, fill=(255, 0, 255), width=10)



# rotate image
polyline_theta = bounds[0]

if isinstance(rotate, float) or isinstance(rotate,int):
    theta = rotate
    
elif isinstance(rotate, str):
    if rotate=='theta':
        theta=polyline_theta
        
    elif rotate=='nearest':
        # nearest
        xy_axis = np.arange(-2*np.pi, 2*np.pi+0.0001, np.pi/2)
        nearest_axis_index = np.argmin(abs(xy_axis - polyline_theta))

        nearest_axis = xy_axis[nearest_axis_index]
        theta = polyline_theta - nearest_axis
        
    elif rotate=='horizontal':
        # horizontal
        x_axis = np.arange(-2*np.pi, 2*np.pi+0.0001, np.pi)
        horizontal_axis_index = np.argmin(abs(x_axis - polyline_theta))

        horizontal_axis = x_axis[horizontal_axis_index]
        theta = polyline_theta - horizontal_axis
        
    elif rotate=='vertical':
        # vertical
        y_axis = np.arange(-3/2*np.pi, 3/2*np.pi+0.0001, np.pi)
        vertical_axis_index = np.argmin(abs(y_axis - polyline_theta))

        vertical_axis = y_axis[vertical_axis_index]
        theta = polyline_theta - vertical_axis
        
    else:
        raise ValueError('rotate is one of the fllowing, False, float, "theta", "nearest", "vertical", "horizontal"')
    
elif rotate==False or rotate==None or polyline_theta==0:  # Do not rotate
    pass  
else:
    raise ValueError('rotate is one of the fllowing, False, float, "theta", "nearest", "vertical", "horizontal"')


if not rotate==False or rotate==None or polyline_theta==0:
    # Rotate image
    polyline_image_rotate = polyline_image.rotate(theta*180/np.pi, expand=True)

    center_point = np.array(polyline_image.size) / 2
    x_c = center_point[0]
    y_c = center_point[1]

    # Use affine transformation, so increase dimension by one.
    dummy_point = np.ones((bounds_pixel.shape[0], 1))

    # Move in parallel to the coordinates with the center as the origin
    bounds_affine = np.concatenate([bounds_pixel, dummy_point], axis=1)
    translation_matrix = np.array([[1,0,-x_c], [0,1,-y_c], [0,0,1]])

    bounds_trans = np.dot(translation_matrix, bounds_affine.T)

    # Rotate from center
    rotation_axis_matrix = self.rotation_axis_matrix_2d(theta, is_affine=True)
    bounds_trans_rotate = np.dot(rotation_axis_matrix, bounds_trans)

    # Move the lower left to the origin again
    after_center_point = np.array(polyline_image_rotate.size) / 2

    x_ca = after_center_point[0]
    y_ca = after_center_point[1]
    after_translation_matrix = np.array([[1,0,x_ca],[0,1,y_ca],[0,0,1]])

    bounds_after = np.dot(after_translation_matrix, bounds_trans_rotate).T
    bounds_after_2d = np.delete(bounds_after,-1,1)

    # use image
    use_image  = polyline_image_rotate
    use_bounds = bounds_after_2d

else:
    use_image  = polyline_image
    use_bounds = bounds_pixel


image_cv = self._pil2cv(use_image)

# bounds is rounded to croped.
# That is because, croped bound must be rounded bound. But drawn bounds is not rounded.
use_bounds = use_bounds.round()
use_bounds = use_bounds.astype(int)


croped, mask, dst, dst2 = self.polygon_crop_cv(image_cv, use_bounds, mode='all')

if crop_mode=='all':
    crop_mode=['croped', 'mask', 'dst', 'dst2']
elif crop_mode!='all' and isinstance(crop_mode,str):
    crop_mode = [crop_mode]
    
if crop_mode==False or crop_mode==None or crop_mode==[]:
    cv2.imwrite(save_path+'nocroped.png', image_cv)

else:        
    if 'croped' in crop_mode:
        ## (1) Crop the bounding rect
        cv2.imwrite(save_path+'croped.png', croped)
    if 'mask' in crop_mode:
        ## (2) make mask
        cv2.imwrite(save_path+'mask.png', mask)
    if 'dst' in crop_mode:
        ## (3) do bit-op
        cv2.imwrite(save_path+'dst.png', dst)
    if 'dst2' in crop_mode:
        ## (4) add the white background
        cv2.imwrite(save_path+'dst2.png', dst2)

# for check, return is exist.
if return_check:
    use_image

    
    
#%%


lpi.concat_tile(bounds, pickup_tiles, \
        file_path='../datasets/', file_extention='.webp', \
        save_path='./', draw_polyline=ls2, draw_bounds=True,  \
        return_check=True, rotate='theta',crop_mode='all')

#%%

#concat_tile_segment
self=LinkingPolylineImage()
pickup_tiles
pickup_tiles_intersection
file_path='../datasets/'
file_extention='.webp'
save_path='./'
draw_polyline=False
draw_bounds=True
crop_mode='dst'


rotate='horizontal'
return_check=False

zoom='self'
TILE_SIZE='self'

"""
A function that collects tiles in bounds and connects the top of 
the satellite image with the polyline.
polyline (shapely.geometry.LineString) : polyline using make bounds.
pickup_tiles (list) : returns for overlappingTile function.
pickup_tiles_intersection (list) : returns for overlappingTileSegment function.


file_path (str) : path for where there are satellite image tiles. ex. '../datasets/'
file_extention (str) : Extension for tile image files. Probably, '.webp'.
save_file_path (str) : path for save image file.

is_polyline (bool) : If True, polyline is drawn.
is_bounds (bool) : If True, bounds is drawn.
crop_mode (list of str) 'croped','mask','dst','dst2' or False: If 'all' 4 all pattern images return.
dst, black back crop, dst2, white back crop.
if False, no croped.

rotate (str or float or False)['theta','nearest', 'vertical', 'horizontal'] : If float, cropped image rotate input angle[rad].
If theta, rotate Angle between start point and end point of polyline,  if nearest, rotate along nearest x or y axis,
if vertical, rotate along the nearest x axis, as well, horizontal y axis.

### Warning ; Pixel coordinate is Y-axis flip. 
The image you see has the coordinates without y-axis flipping, 
but the numbers themselves are in the y-axis flipping coordinate system.
In short, the angle is in the reverse rotation direction as usual.

return_check (bool) : If True, return Pillow image to check.

zoom (int)[0-18] : zoom level.
TILE_SIZE (tuple of int) : The tile size(px). Probably, (256, 256).
"""
        
if zoom=='self' and hasattr(self, 'zoom'):
    zoom = self.zoom
elif zoom=='self' and not hasattr(self, 'zoom'):
    raise KeyError('zoom is not found. Please input zoom or in class instance')

if TILE_SIZE=='self' and hasattr(self, 'TILE_SIZE'):
    TILE_SIZE = self.TILE_SIZE
elif TILE_SIZE=='self' and not hasattr(self, 'TILE_SIZE'):
    raise KeyError('TILE_SIZE is not found. Please input TILE_SIZE or in class instance')

# the last of file_path '/' may or may not be.
pickup_tiles_list = self._pickup_file_search(pickup_tiles, zoom, file_path, file_extention)

# concat image
img_list = [Image.open(pickup_tile) for pickup_tile in pickup_tiles_list]
dummy = self._make_dummy(img_list[0])

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

concated_image = self._concat_tile_blank(arrangement_image)

polyline_image = concated_image.copy()

# draw polyline
if type(draw_polyline)==str:
    draw_polyline = shapely.wkt.loads(draw_polyline)
    
if type(draw_polyline)==shapely.geometry.linestring.LineString:
    polyline = draw_polyline
    polyline_coords = np.array(polyline.coords)
    pixel_coords = np.array(list(map(functools.partial(self.latlng_to_pixel, zoom=zoom,is_round=False), polyline_coords)))

    # change tile coordinate to pixel coordinate, from min point is (0,0).
    pixel_coords[:,0] -= min_x * TILE_SIZE[0]
    pixel_coords[:,1] -= min_y * TILE_SIZE[1]
    pixel_coords_tuple = tuple(map(tuple, pixel_coords))

    draw = ImageDraw.Draw(polyline_image)
    draw.line(pixel_coords_tuple, fill=(255, 255, 0), width=10)
elif draw_polyline==True:
    raise ValueError('draw_polyline is False or shapely.geometry.linestring.LineString')


        
if draw_bounds:
    bounds_pixel_drow = np.concatenate([bounds_pixel, bounds_pixel[[0]]], axis=0)
    bounds_pixel_tuple = tuple(map(tuple, bounds_pixel_drow))

    draw = ImageDraw.Draw(polyline_image)
    draw.line(bounds_pixel_tuple, fill=(255, 0, 255), width=10)

# This function is same to concat_tile function except here.
# It may be necessary to change the common part to another function, but leave it alone.
# -----------------------------------------------------------------------------------------
# draw bounds 
is_bound_tile = np.all(~np.isnan(pickup_tiles_intersection[:,2]), axis=1)

bounds_tile_num = np.where(is_bound_tile)[0]
bounds_position = pickup_tiles_intersection[is_bound_tile][:,2]

len_bounds = len(bounds_tile_num)  # probably 4

bounds_pixel = []
for i in range(len_bounds):
    bound = pt_standard[bounds_tile_num[i]] * TILE_SIZE + bounds_position[i]
    bounds_pixel.append(bound)
    
bounds_pixel = np.array(bounds_pixel)
bounds_pixel = self._to_convex_rectangle(bounds_pixel)


#%%
# rotate image
# polyline angle is not exist, because of bounds is not input, make polyline angle.
if type(draw_polyline)==shapely.geometry.linestring.LineString:
    coords = np.array(polyline.coords)  # all LineString coordinate.
    coords = np.array(self.unit_change(coords, unit=['latlng','pixel'],zoom=zoom, is_round=False))
    start_coord = coords[0]
    end_coord = coords[-1]

else:
    long_boundary = np.argmax([np.linalg.norm(bounds_pixel[0]-bounds_pixel[1]),np.linalg.norm(bounds_pixel[0]-bounds_pixel[-1])])

    # from bounds, make some angle.
    if long_boundary==0: 
        start_coord = bounds_pixel[0]
        end_coord = bounds_pixel[1]
    elif long_boundary==1:
        start_coord = bounds_pixel[0]
        end_coord = bounds_pixel[-1]
    else:
        start_coord = bounds_pixel[0]
        end_coord = bounds_pixel[1]   



#%%

# vector start to end
vec_se = np.array([ end_coord[0]-start_coord[0], end_coord[1]-start_coord[1] ])
# theta from x axis [rad]
polyline_theta = np.arctan2( vec_se[1], vec_se[0] )

#%%

if isinstance(rotate, float) or isinstance(rotate,int):
    theta = rotate
    
elif isinstance(rotate, str):
    if rotate=='theta':
        if type(draw_polyline)==shapely.geometry.linestring.LineString:
            theta=polyline_theta
        else:
            raise ValueError('rotate="theta" is only use type(draw_polyline)=shapely.geometry.linestring.LineString')
        
    elif rotate=='nearest':
        # nearest
        xy_axis = np.arange(-2*np.pi, 2*np.pi+0.0001, np.pi/2)
        nearest_axis_index = np.argmin(abs(xy_axis - polyline_theta))

        nearest_axis = xy_axis[nearest_axis_index]
        theta = polyline_theta - nearest_axis
        
    elif rotate=='horizontal':
        # horizontal
        x_axis = np.arange(-2*np.pi, 2*np.pi+0.0001, np.pi)
        horizontal_axis_index = np.argmin(abs(x_axis - polyline_theta))

        horizontal_axis = x_axis[horizontal_axis_index]
        theta = polyline_theta - horizontal_axis
        
    elif rotate=='vertical':
        # vertical
        y_axis = np.arange(-3/2*np.pi, 3/2*np.pi+0.0001, np.pi)
        vertical_axis_index = np.argmin(abs(y_axis - polyline_theta))

        vertical_axis = y_axis[vertical_axis_index]
        theta = polyline_theta - vertical_axis
        
    else:
        raise ValueError('rotate is one of the fllowing, False, float, "theta", "nearest", "vertical", "horizontal"')
    
elif rotate==False or rotate==None or polyline_theta==0:  # Do not rotate
    pass  
else:
    raise ValueError('rotate is one of the fllowing, False, float, "theta", "nearest", "vertical", "horizontal"')

# ------------------------------------------------------------------------------
# This function is same to concat_tile function except here.
# It may be necessary to change the common part to another function, but leave it alone.


if not rotate==False or rotate==None or polyline_theta==0:
    # Rotate image
    polyline_image_rotate = polyline_image.rotate(theta*180/np.pi, expand=True)

    center_point = np.array(polyline_image.size) / 2
    x_c = center_point[0]
    y_c = center_point[1]

    # Use affine transformation, so increase dimension by one.
    dummy_point = np.ones((bounds_pixel.shape[0], 1))

    # Move in parallel to the coordinates with the center as the origin
    bounds_affine = np.concatenate([bounds_pixel, dummy_point], axis=1)
    translation_matrix = np.array([[1,0,-x_c], [0,1,-y_c], [0,0,1]])

    bounds_trans = np.dot(translation_matrix, bounds_affine.T)

    # Rotate from center
    rotation_axis_matrix = self.rotation_axis_matrix_2d(theta, is_affine=True)
    bounds_trans_rotate = np.dot(rotation_axis_matrix, bounds_trans)

    # Move the lower left to the origin again
    after_center_point = np.array(polyline_image_rotate.size) / 2

    x_ca = after_center_point[0]
    y_ca = after_center_point[1]
    after_translation_matrix = np.array([[1,0,x_ca],[0,1,y_ca],[0,0,1]])

    bounds_after = np.dot(after_translation_matrix, bounds_trans_rotate).T
    bounds_after_2d = np.delete(bounds_after,-1,1)

    # use image
    use_image  = polyline_image_rotate
    use_bounds = bounds_after_2d

else:
    use_image  = polyline_image
    use_bounds = bounds_pixel

        
if draw_bounds:
    bounds_pixel_drow = np.concatenate([bounds_pixel, bounds_pixel[[0]]], axis=0)
    bounds_pixel_tuple = tuple(map(tuple, bounds_pixel_drow))

    draw = ImageDraw.Draw(polyline_image)
    draw.line(bounds_pixel_tuple, fill=(255, 0, 255), width=10)

image_cv = self._pil2cv(use_image)

# bounds is rounded to croped.
# That is because, croped bound must be rounded bound. But drawn bounds is not rounded.
use_bounds = use_bounds.round()
use_bounds = use_bounds.astype(int)


croped, mask, dst, dst2 = self.polygon_crop_cv(image_cv, use_bounds, mode='all')

if crop_mode=='all':
    crop_mode=['croped', 'mask', 'dst', 'dst2']
elif crop_mode!='all' and isinstance(crop_mode,str):
    crop_mode = [crop_mode]
    
if crop_mode==False or crop_mode==None or crop_mode==[]:
    cv2.imwrite(save_path+'nocroped.png', image_cv)

else:        
    if 'croped' in crop_mode:
        ## (1) Crop the bounding rect
        cv2.imwrite(save_path+'croped.png', croped)
    if 'mask' in crop_mode:
        ## (2) make mask
        cv2.imwrite(save_path+'mask.png', mask)
    if 'dst' in crop_mode:
        ## (3) do bit-op
        cv2.imwrite(save_path+'dst.png', dst)
    if 'dst2' in crop_mode:
        ## (4) add the white background
        cv2.imwrite(save_path+'dst2.png', dst2)

# for check, return is exist.

use_image



#%%








lpi.concat_tile_segment(pickup_tiles, pickup_tiles_intersection, \
        file_path='../datasets/', file_extention='.webp', \
        save_path='./', draw_polyline=ls2, draw_bounds=True, crop_mode='dst')