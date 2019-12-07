#%%
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Polygon
import shapely.wkt
import functools
import glob


class TileSegment:
    def __init__(self):
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
            
        self.TILE_SIZE=(256, 256)
    
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
    
    def pixel_to_tile(self,pixel_coordinate, is_remainder=True, TILE_SIZE='self'):
        """pixel coordinate convert to tile coordinate
        tile coordinate explanation:https://www.trail-note.net/tech/coordinate/
        pixel_coordinate (tuple of float) : longitude and latitude coordinate. xy order. For example, (140.087099, 36.104665)
        is_remainder (bool) : Pixel convert to tile, we get remainder. Remainder show coordinate in a tile. If True, return this reminder.
        
        returns : (x_tile, y_tile)
        if is_remainder : (x_tile, y_tile, x_in_tile, y_in_tile)
        """
        if TILE_SIZE=='self' and hasattr(self, 'TILE_SIZE'):
            TILE_SIZE = self.TILE_SIZE
        elif TILE_SIZE=='self' and not hasattr(self, 'TILE_SIZE'):
            raise KeyError('TILE_SIZE is not found. Please input TILE_SIZE or in class instance')
        x_tile = pixel_coordinate[0] // TILE_SIZE[0]
        y_tile = pixel_coordinate[1] // TILE_SIZE[1]
        
        if is_remainder:
            x_in_tile = pixel_coordinate[0] % TILE_SIZE[0]
            y_in_tile = pixel_coordinate[1] % TILE_SIZE[1]
            return x_tile, y_tile, x_in_tile, y_in_tile
        
        return x_tile, y_tile
    
    def latlng_to_tile(self,coordinate, zoom=18, is_remainder=True, is_round=False, TILE_SIZE='self'):
        """atitude and longitude convert to tile coordinate
        tile coordinate explanation:https://www.trail-note.net/tech/coordinate/
        coordinate (tuple of float) : longitude and latitude coordinate. xy order. For example, (140.087099, 36.104665)
        zoom (int)[0-18]: zoom level. Coordinate convert to 2**zoom magnification.
        is_remainder (bool) : Pixel convert to tile, we get remainder. Remainder show coordinate in a tile. If True, return this reminder.
        is_round (bool) : If True, round returns at stage pixel coordinate.
        
        returns : (x_tile, y_tile)
        if is_remainder : (x_tile, y_tile, x_in_tile, y_in_tile)
        """
        pixel_coordinate = self.latlng_to_pixel(coordinate, zoom=zoom, is_round=is_round)
        tile_coordinate = self.pixel_to_tile(pixel_coordinate, is_remainder=is_remainder, TILE_SIZE=TILE_SIZE)
        
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
    
    def tile_to_pixel(self,tile_coordinate, TILE_SIZE='self'):
        """pixel coordinate convert to pixel coordinate.
        pixel coordinate explanation : https://www.trail-note.net/tech/coordinate/
        warning :: pixel coordinate is y axis inversion.
        pixel_coordinate (tuple of float) : pixel coordinate. xy order. For example, (59668600, 26328600)
        zoom (int)[0-18] : zoom level. Coordinate convert to 2**zoom magnification.

        returns : (longitude, latitude)
        """
        if TILE_SIZE=='self' and hasattr(self, 'TILE_SIZE'):
            TILE_SIZE = self.TILE_SIZE
        elif TILE_SIZE=='self' and not hasattr(self, 'TILE_SIZE'):
            raise KeyError('TILE_SIZE is not found. Please input TILE_SIZE or in class instance')

        len_tile_coordinate = len(tile_coordinate)
        if len_tile_coordinate==2:
            is_remainder=False
        elif len_tile_coordinate==4:
            is_remainder=True
        else:
            raise ValueError('tile_coordinate is tuple of (x_tile, y_tile) or (x_tile, y_tile, x_in_tile, y_in_tile)')
        
        x_tile = tile_coordinate[0]
        y_tile = tile_coordinate[1]
        
        if not is_remainder:
            x_pixel = x_tile * TILE_SIZE[0]
            y_pixel = y_tile * TILE_SIZE[1]
        else:
            x_in_tile = tile_coordinate[2]
            y_in_tile = tile_coordinate[3]
            x_pixel = x_tile * TILE_SIZE[0] + x_in_tile
            y_pixel = y_tile * TILE_SIZE[1] + y_in_tile
        
        return x_pixel, y_pixel

    def tile_to_latlng(self,tile_coordinate, zoom=18, TILE_SIZE='self'):
        """atitude and longitude convert to tile coordinate
        tile coordinate explanation:https://www.trail-note.net/tech/coordinate/
        tile_coordinate (tuple of float) : tile coordinate. For example, (233080.0, 102845.0) or (233080.0, 102845.0, 79.98595982044935, 239.97896275669336)
        zoom (int)[0-18]: zoom level. Coordinate convert to 2**zoom magnification.
        
        returns : (x_tile, y_tile)
        """
        pixel_coordinate = self.tile_to_pixel(tile_coordinate, TILE_SIZE=TILE_SIZE)
        tile_coordinate = self.pixel_to_latlng(pixel_coordinate, zoom=zoom)
        
        return tile_coordinate
    
    # ------------- Calculation Minimum Bounding Rectangle aligned xy axis ----------------
    def xy_aligned(self, polyline='self',form='rectangle', minimum=[]):
        """function to Minimum Bounding Rectangle aligned xy axis.
        polyline (shapely.geometry.LineString) : LineString object. If default, use class instance.
        
        form (str)['minmax', 'rectangle'] : If 'minmax', return np.array([[x_min, y_min], [x_max, y_max]]).
        If 'rectangle', return [theta(=0), np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]]) ]
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

        if form=='rectangle':
            # point order is counterclockwise.
            # [theta from x axis, np.array([4 points of spuare])]
            bounds = [0, np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]]) ]
        
        return bounds
    
    @staticmethod
    def rotation_axis_matrix_2d(theta):
        return np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
    
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
        
        rotation_axis_matrix = self.rotation_axis_matrix_2d(theta)
        # Coordinate transformation using rotation.
        coords_trans = np.dot(rotation_axis_matrix, coords.T).T

        # get_bounds
        bounds_trans = self.xy_aligned( LineString(coords_trans),form='rectangle', minimum=minimum)
        # reverse coordinate transformation
        rotation_axis_matrix_rev = self.rotation_axis_matrix_2d(-theta)
        bounds_ = np.dot(rotation_axis_matrix_rev, bounds_trans[1].T).T
        bounds = [theta, bounds_]
        
        return bounds
    
    @staticmethod
    def inpolygon(point, polygon):
        ''' 
        Judgment if point is included in polygon by Crossing Number Algorithm.
        point (list or tuple or np.array) : Coordinate to judge inside / outside. ex : [x,y],(x,y) , np.array([x,y])
        polygon (np.array, ndim=2) : Vertex coordinates of polygon. ex : np.array([[x1,y1], [x2,y2], [x3,y3], ...]).
        **end_point is not closing.**
        '''
        point_x = point[0]
        point_y = point[1]
        len_polygon = polygon.shape[0]
        if polygon.shape[1]!=2:
            raise KeyError('polygon is np.array([[x1,y1], [x2,y2], [x3,y3], ...])')
        x = polygon[:,0]
        y = polygon[:,1]
        
        inside = False
        for i1 in range(len_polygon): 
            i2 = (i1+1)%len_polygon
            if min(x[i1], x[i2]) < point_x < max(x[i1], x[i2]):
                #a = (y[i2]-y[i1])/(x[i2]-x[i1])
                #b = y[i1] - a*x[i1]
                #dy = a*point_x+b - point_y
                #if dy >= 0:
                if (y[i1] + (y[i2]-y[i1])/(x[i2]-x[i1])*(point_x-x[i1]) - point_y) > 0:
                    inside = not inside

        return inside
    def overlappingTiles(self, bounds, zoom=18, filepath=False, file_extention='.webp',TILE_SIZE='self'):
        """
        bounds (list) : input [theta, np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min, y_max]])] by xy_aligned or terminal node aligned.
        zoom (int)[0-18] : zoom level.
        filepath (str of False) : tiles database path. If path is --/--/--/zoom/x/y, --/--/--/.
        Don't need it, if not False, check whether file exists.
        file_extention (str) : If use filepath, specify file extention.
        """
        ### タイルのどの点もboundsに含まれないが、タイルの中に点があるケースを拾えていない。 ###
        # boundsの各点に対して、その場所のタイルに点が含まれるかを判定してやればいい。しかしxy_alignedは問題なし。
        
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

        # for make searching point, Make roughly tiles points
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
        # Tiles are represented by 4 points number
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

        pickup_tiles = points[tiles[is_in_bounds,0]] / TILE_SIZE
        pickup_tiles = pickup_tiles.astype(int)

        pickup_tiles_list = []
        for pit in pickup_tiles:
            x = pit[0]
            y = pit[1]
            path = filepath +str(x) + '/'+str(y) + file_extention
            pickup_tiles_list.append(path)
        
        if filepath:
            # all database tiles.
            tilesfile_list = glob.glob(filepath+'*/*')
            isnot_exist_files = set(pickup_tiles_list) - set(tilesfile_list)

            # if pickup_file is not exist, raise error
            if not isnot_exist_files==set():
                raise ValueError(str(isnot_exist_files)+' is not exist')
        
        return pickup_tiles_list     
    
    @staticmethod     
    def is_intersected_ls(a1, a2, b1, b2):
        """
        Judgment if line and line intersect.
        a1 (np.array) : one point of line a. ex. np.array([0,0])
        a2 (np.array) : another point of line a. ex. np.array([1,1])
        b1, b2(np.array) : points of line b.
        """
        EPS = 0.000001
        cross_b_and_a = np.cross(a2-a1, b1-a1) * np.cross(a2-a1, b2-a1)
        cross_a_and_b = np.cross(b2-b1, a1-b1) * np.cross(b2-b1, a2-b1)
        if cross_b_and_a<EPS and cross_a_and_b<EPS:
            return True
        else:
            return False

    @staticmethod
    def intersection_ls(a1, a2, b1, b2):
        """
        Find intersection of line and line .
        a1 (np.array) : one point of line a. ex. np.array([0,0])
        a2 (np.array) : another point of line a. ex. np.array([1,1])
        b1, b2(np.array) : points of line b.
        """
        b = b2-b1
        d1 = abs(np.cross(b, a1-b1))
        d2 = abs(np.cross(b, a2-b1))
        t = d1 / (d1 + d2)

        return a1 + (a2-a1) * t
    
    def overlappingTileSegments(self, bounds, zoom=18, TILE_SIZE='self'):


        if TILE_SIZE=='self' and hasattr(self, 'TILE_SIZE'):
            TILE_SIZE = self.TILE_SIZE
        elif TILE_SIZE=='self' and not hasattr(self, 'TILE_SIZE'):
            raise KeyError('TILE_SIZE is not found. Please input TILE_SIZE or in class instance')


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

        return pickup_tiles, pickup_tiles_intersection
    

