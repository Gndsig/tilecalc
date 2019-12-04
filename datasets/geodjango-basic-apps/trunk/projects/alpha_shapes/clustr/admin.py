# admin.py

from django.conf import settings # needed if we use the GOOGLE_MAPS_API_KEY from settings
from django.contrib import admin
from django.contrib.gis.admin import OSMGeoAdmin, GeoModelAdmin
from clustr.models import AlphaShapes

# Import the Databrowse app so we can register our models to display via the Databrowse
from django.contrib import databrowse
databrowse.site.register(AlphaShapes)

USE_GOOGLE_TERRAIN_TILES = False

class AlphaShapesAdmin(OSMGeoAdmin):
    # Standard Django Admin Options
    #list_display = ('name','pop2005','region','subregion','geometry',)
    #list_editable = ('geometry',)
    #search_fields = ('name',)
    #list_per_page = 4
    #ordering = ('name',)
    #list_filter = ('region','subregion',)
    #save_as = True
    #search_fields = ['name','iso2','iso3','subregion','region']
    #list_select_related = True
    if USE_GOOGLE_TERRAIN_TILES:
      map_template = 'gis/admin/google.html'
      extra_js = ['http://openstreetmap.org/openlayers/OpenStreetMap.js', 'http://maps.google.com/maps?file=api&amp;v=2&amp;key=%s' % settings.GOOGLE_MAPS_API_KEY]
    else:
      pass # defaults to OSMGeoAdmin presets of OpenStreetMap tiles

    # Default GeoDjango OpenLayers map options
    # Uncomment and modify as desired
    # To learn more about this jargon visit:
    # www.openlayers.org
    
    #default_lon = 0
    #default_lat = 0
    #default_zoom = 4
    #display_wkt = False
    #display_srid = False
    #extra_js = []
    #num_zoom = 18
    #max_zoom = False
    #min_zoom = False
    #units = False
    #max_resolution = False
    #max_extent = False
    #modifiable = True
    #mouse_position = True
    #scale_text = True
    #layerswitcher = True
    scrollable = False
    #admin_media_prefix = settings.ADMIN_MEDIA_PREFIX
    map_width = 400
    map_height = 325
    #map_srid = 4326
    #map_template = 'gis/admin/openlayers.html'
    #openlayers_url = 'http://openlayers.org/api/2.6/OpenLayers.js'
    #wms_url = 'http://labs.metacarta.com/wms/vmap0'
    #wms_layer = 'basic'
    #wms_name = 'OpenLayers WMS'
    #debug = False
    #widget = OpenLayersWidget

# Finally, with these options set now register the model
# associating the Options with the actual model
admin.site.register(AlphaShapes,AlphaShapesAdmin)