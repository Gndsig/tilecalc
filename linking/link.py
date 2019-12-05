#%%
import numpy as np
from shapely.geometry import LineString
import shapely.wkt

class TileSegment:
    def __init__(self):
        # type judgment
        
        pass
    
    

class LinkingPolylineImage:
    
    # constructor for WKT
    def __init__(self, wkt_polyline):
        """
        wkt_polyline (str or shapely.geometry.LineString) : WKT string or LineString object.
        For example, 'LINESTRING(139.07730102539062 36.00022956359412, 139.0814208984375 35.98022880246021)'
        ,or shapely.geometry.LineString([(139.07455444335938, 35.9913409624497), (139.07730102539062, 35.99356320664446)])
        """
        if isinstance(wkt_polyline, str):
            self.polyline = shapely.wkt.loads(wkt_polyline)
        else:
            self.polyline = wkt_polyline
    
    # ------------- convert latitude and longitude to XYZtile or vice versa ----------------
    @classmethod
    def latlng_to_pixel(self,coordinate, zoom=18, is_round=False):
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
    
    @classmethod
    def pixel_to_tile(self,pixel_coordinate, is_segment=False):
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
    @classmethod
    def latlng_to_tile(self,coordinate, zoom=18, is_segment=True, is_round=False):
        """atitude and longitude convert to tile coordinate
        tile coordinate explanation:https://www.trail-note.net/tech/coordinate/
        coordinate (tuple of float) : longitude and latitude coordinate. xy order. For example, (140.087099, 36.104665)
        zoom (int)[0-18]: zoom level. Coordinate convert to 2**zoom magnification.
        is_segment (bool) : Pixel convert to tile, we get remainder. Remainder show coordinate in a tile. If True, return this reminder.
        is_round (bool) : If True, round returns at stage pixel coordinate.
        
        returns : (x_tile, y_tile)
        if is_segment : (x_tile, y_tile, x_in_tile, y_in_tile)
        """
        pixel_coordinate = self.latlng_to_pixel(coordinate, zoom=zoom, is_round=is_round)
        tile_coordinate = self.pixel_to_tile(pixel_coordinate, is_segment=is_segment)
        
        return tile_coordinate
    
    @classmethod
    def pixel_to_latlng(self, pixel_coordinate, zoom=18):
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
    @classmethod
    def tile_to_pixel(self,tile_coordinate):
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
    @classmethod
    def tile_to_latlng(self,tile_coordinate, zoom=18):
        """atitude and longitude convert to tile coordinate
        tile coordinate explanation:https://www.trail-note.net/tech/coordinate/
        tile_coordinate (tuple of float) : tile coordinate. For example, (233080.0, 102845.0) or (233080.0, 102845.0, 79.98595982044935, 239.97896275669336)
        zoom (int)[0-18]: zoom level. Coordinate convert to 2**zoom magnification.
        
        returns : (x_tile, y_tile)
        """
        pixel_coordinate = self.tile_to_pixel(tile_coordinate)
        tile_coordinate = self.pixel_to_latlng(pixel_coordinate, zoom=zoom)
        
        return tile_coordinate
    
    
    # ------------- Calculation Minimum Bounding Rectangle aligned xy axis ----------------
    @classmethod
    def xy_aligned(self, polyline='self',form='general', minimum=[]):
        """function to Minimum Bounding Rectangle aligned xy axis.
        polyline (shapely.geometry.LineString) : LineString object. If default, use class instance.
        
        form (str)['bounds', 'general'] : If 'bounds', return np.array([[x_min, y_min], [x_max, y_max]]).
        If 'general', return [theta(=0), np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]]) ]
        Theta is angle[rad] from x axis, this case is 0.
        
        minimum (list of [x_min,y_min]) : Minimum width of Minimum Bounding Rectangle.
        If width < minimum, width=minimum and Rectangle is helf width from center point.
        """
        if polyline=='self' and hasattr(self, 'polyline'):
            polyline = self.polyline
        elif polyline=='self' and not hasattr(self, 'polyline'):
            raise KeyError('polyline is not found. Please input polyline or in class instance')
        bounds = polyline.bounds
        x_min = bounds[0]
        y_min = bounds[1]
        x_max = bounds[2]
        y_max = bounds[3]

        center_point = ( (x_min+x_max)/2.0 , (y_min+y_max)/2.0 )

        width = [ x_max-x_min, y_max-y_min]

        if not(minimum==() or minimum==[] or minimum==None):
            # correct mimimum bounds
            try:
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

        if form=='general':
            # point order is counterclockwise.
            # [theta from x axis, np.array([4 points of spuare])]
            bounds = [0, np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]]) ]
        
        return bounds
    
    @classmethod
    def terminal_node_aligned(self, polyline='self', minimum=[]):
        """function to Minimum Bounding Rectangle aligned vector start to end.
        polyline (shapely.geometry.LineString) : LineString object. If default, use class instance.
        minimum (list of [x_min,y_min]) : Minimum width of Minimum Bounding Rectangle.
        If width < minimum, width=minimum and Rectangle is helf width from center point.
        
        returns : [theta, np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]]) ]
        Theta is angle[rad] vector start to end from x axis.
        """

        if polyline=='self' and hasattr(self, 'polyline'):
            polyline = self.polyline
        elif polyline=='self' and not hasattr(self, 'polyline'):
            raise KeyError('polyline is not found. Please input polyline or in class instance')
        
        coords = np.array(polyline.coords)  # all LineString coordinate.

        start_coord = coords[0]
        end_coord = coords[-1]
        # vector start to end
        vec_se = np.array([ end_coord[0]-start_coord[0], end_coord[1]-start_coord[1] ])
        # theta from x axis [rad]
        theta = np.arctan2( vec_se[1], vec_se[0] )

        def rotation_axis_matrix_2d(theta):
            return np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        
        rotation_axis_matrix = rotation_axis_matrix_2d(theta)
        # Coordinate transformation using rotation.
        coords_trans = np.dot(rotation_axis_matrix, coords.T).T

        # get_bounds
        bounds_trans = self.xy_aligned( LineString(coords_trans),form='general', minimum=minimum)
        rotation_axis_matrix_rev = rotation_axis_matrix_2d(-theta)
        bounds_ = np.dot(rotation_axis_matrix_rev, bounds_trans[1].T).T
        bounds = [theta, bounds_]
        
        return bounds
    
    
    
    def overlappingTiles(self, filepath, polyline='self',zoom=18):
            # XY coordinate of XYZTile covering WKTPolyline
        if polyline=='self' and hasattr(self, 'polyline'):
            polyline = self.polyline
        elif polyline=='self' and not hasattr(self, 'polyline'):
            raise KeyError('polyline is not found. Please input polyline or in class instance')
        
        
    
    def overlappingTileSegments(self, zoom, mbr_type):
        pass
    


# %%
zoom=18
coordinate=(140.087099, 36.104665)

line_sample1 = 'LINESTRING(139.07730102539062 36.00022956359412, 139.0814208984375 35.98022880246021)'
ls1 = shapely.wkt.loads(line_sample1)
ls2 = LineString([(139.07730102539062,36.00022956359412),(139.0814208984375,35.98022880246021),(139.07867431640625, 35.99356320664446),(139.07455444335938, 35.9913409624497)])


#%%
# sample
print(ls2.area)
print(ls2.bounds)
print(ls2.length)
print(ls2.geom_type)
print(ls2.distance)
list(ls2.coords)




tile = LinkingPolylineTile.latlng_to_tile(coordinate, zoom, is_segment=True)
tile
# %%
coord = LinkingPolylineTile.tile_to_latlng(tile)
coord

#%%
tile

#%%
coord