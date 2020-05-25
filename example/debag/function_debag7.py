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
pickup_tiles = lpi.overlappingTiles(bounds, zoom=zoom)
pickup_tiles

# %%
# return xy tile coordinate array and where intersection is in tile.
#pickup_tiles, pickup_tiles_intersection = lpi.overlappingTileSegments(bounds, zoom=zoom)
#print(pickup_tiles)
#print(pickup_tiles_intersection)

# pickup_tiles_intersection as np.array([[x1, y1], [x2, y2],...]), 
# np.array([[x11,y11],[x12,y12],[x13,y13]], ...). x11, y11 and x12, y12 is intersections of bounds lines and tiles lines.
# x13, y13 is bounds corner points in the tiles. If intersections or corner points are not exist, it is in np.nan.
# elif filepath='anypath', return pickup_tiles, pickup_tiles_intersection, pickup_tiles_list. pickup_tile_list as ['filepath/zoom/x1/y1',...].

#%%
lpi.concat_tile(polyline, bounds, pickup_tiles, file_path='../datasets/', save_path='./images/', is_polyline=True, is_bounds=False, crop_mode='dst')

#%%
# concat_tile
self = LinkingPolylineImage()
polyline
bounds
pickup_tiles
file_path = '../datasets/'
file_extention='.webp'
save_path='./'
mode='croped'
zoom='self'
TILE_SIZE='self'

a="""
A function that collects tiles in bounds and connects the top of 
the satellite image with the polyline.
polyline (shapely.geometry.LineString) : polyline using make bounds.
bounds (list) : returns for xy_aligned or terminal_node_aligned.
pickup_tiles (list) : returns for overlappingTile function.

file_path (str) : path for where there are satellite image tiles. ex. '../datasets/'
file_extention (str) : Extension for tile image files. Probably, '.webp'.
save_file_path (str) : path for save image file.
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

# draw polyline
polyline_coords = np.array(polyline.coords)
pixel_coords = np.array(list(map(functools.partial(self.latlng_to_pixel, zoom=zoom,is_round=False), polyline_coords)))

pixel_coords[:,0] -= min_x * TILE_SIZE[0]
pixel_coords[:,1] -= min_y * TILE_SIZE[1]
pixel_coords_tuple = tuple(map(tuple, pixel_coords))

polyline_image = concated_image.copy()
draw = ImageDraw.Draw(polyline_image)
draw.line(pixel_coords_tuple, fill=(255, 255, 0), width=10)
base_and_polyline = polyline_image.copy()

# draw bounds
bounds_pixel = bounds[1].copy()
bounds_pixel[:,0] -= min_x * TILE_SIZE[0]
bounds_pixel[:,1] -= min_y * TILE_SIZE[1]
bounds_pixel_drow = np.concatenate([bounds_pixel, bounds_pixel[[0]]], axis=0)
bounds_pixel_tuple = tuple(map(tuple, bounds_pixel_drow))

draw = ImageDraw.Draw(polyline_image)
draw.line(bounds_pixel_tuple, fill=(255, 0, 255), width=10)
base_and_polyline_and_bounds = polyline_image.copy()


image_cv = self._pil2cv(polyline_image)
#image_cv = pil2cv(concated_image)
bounds_pixel = bounds_pixel.astype(int)


croped, mask, dst, dst2 = self.polygon_crop_cv(image_cv, bounds_pixel, mode='all')

if mode=='all':
    mode=['croped', 'mask', 'dst', 'dst2']
elif mode!='all' and isinstance(mode,str):
    mode = [mode]
    
if 'croped' in mode:
    ## (1) Crop the bounding rect
    cv2.imwrite(save_path+"croped.png", croped)
    print(1)
if 'mask' in mode:
    ## (2) make mask
    cv2.imwrite(save_path+"mask.png", mask)
if 'dst' in mode:
    ## (3) do bit-op
    cv2.imwrite(save_path+"dst.png", dst)
if 'dst2' in mode:
    ## (4) add the white background
    cv2.imwrite(save_path+"dst2.png", dst2)

#%%
bounds

#%%
bounds_pixel

#%%

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


croped, mask, dst, dst2 = polygon_crop_cv(image_cv, bounds_pixel)

#%%
croped

#%%
savepath = './images/'
cv2.imwrite(savepath+"croped.png", croped)
cv2.imwrite(savepath+"mask.png", mask)

cv2.imwrite(savepath+"dst.png", dst)
cv2.imwrite(savepath+"dst2.png", dst2)


# for check, return is exist.
base_and_polyline_and_bounds













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