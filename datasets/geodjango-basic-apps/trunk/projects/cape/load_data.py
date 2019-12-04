#!/usr/bin/python

"""Utility for adding the Ward layermapping from data/ to the cape.lionhead
   project."""

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'settings'
from psycopg2 import IntegrityError

from django.contrib.gis.utils import mapping, LayerMapping, add_postgis_srs
from lionshead.models import Ward

try:
    add_postgis_srs(900913)
except IntegrityError:
    print "The Google Spherical Mercator projection, or a projection with srid 900913, already exists, skipping insert"

wards = 'data/wards_4326.shp'
wlayer =  LayerMapping(Ward, wards,
    mapping(wards, geom_name='geometry',multi_geom=True), 
    encoding='Latin1')
wlayer.save(verbose=True, progress=True)
