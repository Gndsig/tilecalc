#%%
import numpy as np
from shapely.geometry import LineString
import shapely.wkt

#%%
#import sys
#sys.path.append('.../')
#from ...linking.link import LinkingPolylineImage

#%%
line_sample1 = 'LINESTRING(139.07730102539062 36.00022956359412, 139.0814208984375 35.98022880246021)'
ls1 = shapely.wkt.loads(line_sample1)
ls2 = LineString([(139.07455444335938, 35.9913409624497), (139.07730102539062, 35.99356320664446),(139.07867431640625,35.99522880246021),(139.0814208984375,36.00022956359412)])

#%%
ls2

#%%
# sample
print(ls2.area)
print(ls2.bounds)
print(ls2.length)
print(ls2.geom_type)
print(ls2.distance)
list(ls2.coords)

#%%

#%%
#def xy_aligned(self, minimum=None):

self = LinkingPolylineImage(ls2)
#form='bounds'
form='general'
#minimum=()
minimum=(0.010, 0.010)

# %%
bounds = self.polyline.bounds
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
    bounds = (x_min, y_min, x_max, y_max)

if form=='general':
    # point order is counterclockwise.
    # (theta, (4 points of spuare))
    bounds = (0, ((x_min,y_min), (x_max,y_min), (x_max,y_max),(x_min, y_max)) )




#%%
#def terminal_node_aligned(self, minimum=None):

self = LinkingPolylineImage(ls2)
#form='bounds'
polyline=ls2
form='general'
minimum=()
#minimum=(0.010, 0.010)

#%%
if polyline=='self' and hasattr(self, 'polyline'):
    polyline = self.polyline
elif polyline=='self' and not hasattr(self, 'polyline'):
    raise KeyError('polyline is not found. Please input polyline or in class instance')

coords = np.array(polyline.coords)

start_coord = coords[0]
end_coord = coords[-1]
# vector start to end
vec_se = np.array([ end_coord[0]-start_coord[0], end_coord[1]-start_coord[1] ])
# theta from x axis [rad]
theta = np.arctan2( vec_se[1], vec_se[0] )

def rotation_axis_matrix_2d(theta):
    return np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
rotation_axis_matrix = rotation_axis_matrix_2d(theta)

# Rotation coordinate transformation
coords_trans = np.dot(rotation_axis_matrix, coords.T).T

bounds_trans = self.xy_aligned( LineString(coords_trans),form='general', minimum=minimum)
rotation_axis_matrix_rev = rotation_axis_matrix_2d(-theta)
bounds_ = np.dot(rotation_axis_matrix_rev, bounds_trans[1].T).T

bounds = [theta, bounds_]

#%%
from shapely.geometry import Polygon
Polygon(bounds_)