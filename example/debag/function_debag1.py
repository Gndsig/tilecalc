#%%
import numpy as np
import shapely


#%%
def latlng_to_pixel(coordinate, zoom=18, is_round=False):
    """latitude and longitude convert to pixel coordinate
    pixel coordinate explanation:https://www.trail-note.net/tech/coordinate/
    zoom (int)[0-18]: zoom level. Coordinate convert to 2**zoom magnification.
    coordinate (tuple of float) : longitude and latitude coordinate. xy order. For example, (140.087099, 36.104665)
    is_round (bool) : If True, round returns coordinate
    
    returns : (x_pixel, y_pixel)
    """
    latitude = coordinate[0]
    longtitude = coordinate[1]
    x_pixel = 2**(zoom+7)*(latitude/180 + 1)
    L=85.05112878
    y_pixel = 2**(zoom+7)/np.pi * (-np.arctanh(np.sin(np.pi/180*longtitude))+np.arctanh(np.sin(np.pi/180*L)))
    
    if is_round:  
        x_pixel = round(x_pixel)
        y_pixel = round(y_pixel)
    
    return x_pixel, y_pixel

def pixel_to_tile(pixel_coordinate, is_segment=False):
    """pixel coordinate convert to tile coordinate
    tile coordinate explanation:https://www.trail-note.net/tech/coordinate/
    pixel_coordinate (tuple of float) : longitude and latitude coordinate. xy order. For example, (140.087099, 36.104665)
    is_segment (bool) : Pixel convert to tile, we get remainder. Remainder show coordinate in a tile. If True, return this reminder.
    
    returns : (x_tile, y_tile)
    if is_segment : (x_tile, y_tile, x_in_tile, y_in_tile)
    """
    TILE_SIZE = 256
    x_tile = pixel_coordinate[0] // TILE_SIZE
    y_tile = pixel_coordinate[1] // TILE_SIZE
    
    if is_segment:
        x_in_tile = pixel_coordinate[0] % TILE_SIZE
        y_in_tile = pixel_coordinate[1] % TILE_SIZE
        return x_tile, y_tile, x_in_tile, y_in_tile
    
    return x_tile, y_tile
   
def latlng_to_tile(coordinate, zoom=18, is_segment=False, is_round=False):
    """atitude and longitude convert to tile coordinate
    tile coordinate explanation:https://www.trail-note.net/tech/coordinate/
    coordinate (tuple of float) : longitude and latitude coordinate. xy order. For example, (140.087099, 36.104665)
    zoom (int)[0-18]: zoom level. Coordinate convert to 2**zoom magnification.
    is_segment (bool) : Pixel convert to tile, we get remainder. Remainder show coordinate in a tile. If True, return this reminder.
    is_round (bool) : If True, round returns at stage pixel coordinate.
    
    returns : (x_tile, y_tile)
    if is_segment : (x_tile, y_tile, x_in_tile, y_in_tile)
    """
    pixel_coordinate = latlng_to_pixel(coordinate, zoom=zoom, is_round=is_round)
    tile_coordinate = pixel_to_tile(pixel_coordinate, is_segment=is_segment)
    
    return tile_coordinate

 
 
 
 
 
#%%
def pixel_to_latlng(pixel_coordinate, zoom=18):
    """pixel coordinate convert to latitude and longitude
    pixel coordinate explanation : https://www.trail-note.net/tech/coordinate/
    pixel_coordinate (tuple of float) : pixel coordinate. xy order. For example, (59668600, 26328600)
    zoom (int)[0-18] : zoom level. Coordinate convert to 2**zoom magnification.

    returns : (longitude, latitude)
    """
    x_pixel = pixel_coordinate[0]
    y_pixel = pixel_coordinate[1]
    L=85.05112878
    longitude = 180 * (x_pixel/(2**(zoom+7)) - 1)
    latitude = 180/np.pi * np.arcsin(np.tanh( -np.pi/(2**(zoom+7))*y_pixel + np.arctanh(np.sin(np.pi/180*L))))
    
    return longitude, latitude

def tile_to_pixel(tile_coordinate):
    """pixel coordinate convert to pixel coordinate
    pixel coordinate explanation : https://www.trail-note.net/tech/coordinate/
    pixel_coordinate (tuple of float) : pixel coordinate. xy order. For example, (59668600, 26328600)
    zoom (int)[0-18] : zoom level. Coordinate convert to 2**zoom magnification.

    returns : (longitude, latitude)
    """
    TILE_SIZE = 256

    len_tile_coordinate = len(tile_coordinate)
    if len_tile_coordinate==2:
        is_segment=False
    elif len_tile_coordinate==4:
        is_segment=True
    else:
        raise ValueError('tile_coordinate is tuple of (x_tile, y_tile) or (x_tile, y_tile, x_in_tile, y_in_tile)')
    
    x_tile = tile_coordinate[0]
    y_tile = tile_coordinate[1]
    
    if not is_segment:
        x_pixel = x_tile * TILE_SIZE
        y_pixel = y_tile * TILE_SIZE
    else:
        x_in_tile = tile_coordinate[2]
        y_in_tile = tile_coordinate[3]
        x_pixel = x_tile * TILE_SIZE + x_in_tile
        y_pixel = y_tile * TILE_SIZE + y_in_tile
    
    return x_pixel, y_pixel

def tile_to_latlng(tile_coordinate, zoom=18):
    """atitude and longitude convert to tile coordinate
    tile coordinate explanation:https://www.trail-note.net/tech/coordinate/
    tile_coordinate (tuple of float) : tile coordinate. For example, (233080.0, 102845.0) or (233080.0, 102845.0, 79.98595982044935, 239.97896275669336)
    zoom (int)[0-18]: zoom level. Coordinate convert to 2**zoom magnification.
    
    returns : (x_tile, y_tile)
    """
    pixel_coordinate = tile_to_pixel(tile_coordinate)
    tile_coordinate = pixel_to_latlng(pixel_coordinate, zoom=zoom)
    
    return tile_coordinate
    
#%%
zoom=18
coordinate=(140.087099, 36.104665)

#%%
tile = latlng_to_tile(coordinate, zoom=zoom, is_segment=True, is_round=False)

coord = tile_to_latlng(tile)


# %%
tile

# %%
coord

# %%
tile = (232345, 102940)
image_coord = tile_to_latlng(tile)
image_coord
# %%
tile2 = (232348, 102958)
image_coord2 = tile_to_latlng(tile2)

image_coord2

# %%
tile3 = (232346, 102946)
image_coord3 = tile_to_latlng(tile3)

image_coord3

# %%
tile4 = (232343, 102948)
image_coord4 = tile_to_latlng(tile4)

image_coord4



