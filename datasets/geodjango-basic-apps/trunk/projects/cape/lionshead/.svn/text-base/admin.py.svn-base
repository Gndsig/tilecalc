from django.contrib.gis import admin
from models import *

class InterestingLocationAdmin(admin.GeoModelAdmin):
    """
    Determines how we will display the InterestingLocation model within the Admin App.  
    We're using the OSGMGeoAdmin subclass which will give us a map with OpenStreetMap data in
    the background, and will also reproject our geographic point data to the appropriate mercator projection
    To see all the GeoDjango options dive into the source code available at:
    http://code.djangoproject.com/browser/django/trunk/django/contrib/gis/admin/options.py
    """
    list_display = ('name','interestingness')
    list_filter = ('name','interestingness',)
    fieldsets = (
      ('Location Attributes', {'fields': (('name','interestingness'))}),
      ('Editable Map View', {'fields': ('geometry',)}),
    )

    # Default GeoDjango OpenLayers map options
    scrollable = False
    map_width = 700
    map_height = 325
    openlayers_url = '/static/openlayers/lib/OpenLayers.js'


class WardAdmin(admin.OSMGeoAdmin):
    list_display = ('cllr','ward')
    list_filter = ('cllr','ward',)
    fieldsets = (
      ('Location Attributes', {'fields': (('cllr','ward'))}),
      ('Editable Map View', {'fields': ('geometry',)}),
    )

    # Default GeoDjango OpenLayers map options
    scrollable = False
    map_width = 700
    map_height = 325
    openlayers_url = '/static/openlayers/lib/OpenLayers.js'

# Register our model and admin options with the admin site
admin.site.register(InterestingLocation, InterestingLocationAdmin)
admin.site.register(Ward, WardAdmin)

