# -*- coding: utf8 -*-

## NOTE: this script written by humanhistory (find him in #geodjango irc)
## and modified by dane springmeyer
## this script is BETA and under development - get in touch for updated versions...
## don't expect it to work very well!

import sys
import logging
from lxml import etree
from datetime import datetime
from psycopg2 import IntegrityError
 
from django.core.management import setup_environ
import settings
setup_environ(settings)
from django.contrib.gis.geos import GEOSGeometry, fromstr
from django.contrib.gis.utils import add_postgis_srs
from django.db import transaction
from clustr.models import AlphaShapes


# Download this data from:
# http://www.flickr.com/services/shapefiles/1.0/
FLICKR_SHAPES_XML_FILE = "clustr/flickr_shapefiles_public_dataset_1.0.xml"


# why not try to toss in the gmerc projection while we are here...
try:
    add_postgis_srs(900913)
except IntegrityError:
    print "The Google Spherical Mercator projection, or a projection with srid 900913, already exists, skipping insert"
 
logging.basicConfig(level=logging.ERROR, 
    datefmt="%Y-%m-%d %H:%M:%S", 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
 
def sort_list_by_obj_attr(seq, attr):
    """
    Borrowed from ActiveState recipe #52230, adjusted for using lxml elements
    """
    intermed = [ (seq[i].get(attr), i, seq[i]) for i in xrange(len(seq)) ]
    intermed.sort()
    return [ tup[-1] for tup in intermed ]
    
 
###########################
# parse the flickr data set
###########################
 
# remove old AlphaShapes
logging.debug("Removing old AlphaShapes objects")
AlphaShapes.objects.filter().delete()
logging.debug("Removed old AlphaShapes objects")
 
context = etree.iterparse(FLICKR_SHAPES_XML_FILE, tag="{x-urn:flickr:}place")
 
for event, elem in context:
    # extract basic info
    place_label = elem.get('label').encode('utf8')
    woe_id = elem.get('woe_id')
    place_id = elem.get('place_id')
    place_type = elem.get('place_type').encode('utf8')
    place_type_id = elem.get('place_type_id')
    
    logging.debug("Parsing %s" % place_label)
 
    # extract shapes
    shapes = elem.xpath("//t:shape", namespaces={'t': "x-urn:flickr:"})
    
    if not shapes:
        logging.error("No shapes defined for %s" % place_label)
        continue
    
    # order by -created, get the most recently generated shape
    sorted_shapes = sort_list_by_obj_attr(shapes, "created")
    sorted_shapes.reverse()
    latest_shape = sorted_shapes[0]
    
    latest_shape_created_date = datetime.fromtimestamp(float(latest_shape.get('created')))
    
    logging.info("%s created on %s" % (place_label, latest_shape_created_date))
    
    # get all polylines
    polylines = latest_shape.xpath("//t:polyline", namespaces={'t': "x-urn:flickr:"})
    
    if not polylines:
        logging.error("No polylines for %s" % place_label)
        continue
    
    p = polylines[0]
    
    # translate pairs for `POLYGON` compatibility
    # 45.289924621582,-64.774787902832 45.294815063477,-64.777793884277
    # becomes:
    # -64.774787902832 45.289924621582, -64.777793884277 45.294815063477
    raw_polyline = p.text
    
    if not raw_polyline:
        logging.error("No polyline data for %s" % place_label)
        continue
    
    raw_pairs = raw_polyline.split(' ')
 
    fixed_pairs = []
    
    for pair in raw_pairs:
        pair = pair.split(',')
        if len(pair) != 2:
            logging.error("Invalid pair for %s: %s" % (
                place_label, str(pair)))
        else:
            lat, lng = pair
        fixed_pairs.append("%s %s" % (lng, lat))
 
    polyline = ','.join(fixed_pairs)
    
    # create new AlphaShapes
    try:
        place = AlphaShapes.objects.create(
            woe_id = woe_id,
            place_id = place_id,
            place_type = place_type,
            place_type_id = place_type_id,
            label = place_label,
            created = latest_shape_created_date,
            poly = GEOSGeometry('POLYGON((%s))' % polyline))
        #print '%s loaded ' % woe_id
    except:
        logging.error("Could not create %s: %s" % (
            place_label, sys.exc_info()[1]))
        transaction.rollback_unless_managed()
        continue
        
    logging.debug("Created %s (%s): pk: %s" % (
        place_label, place_type, place.pk))
        
    # It's safe to call clear() here because no descendants will be accessed
    elem.clear()
    
    # Also eliminate now-empty references from the root node to <Title> 
    while elem.getprevious() is not None:
        del elem.getparent()[0]