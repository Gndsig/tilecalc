# views.py

import os
import mapnik
from django.conf import settings
from django.shortcuts import render_to_response
from clustr.shortcuts import render_to_geojson
from django.http import HttpResponse, Http404
from clustr.models import AlphaShapes

from django.contrib.gis.geos import *

MAP_CACHE = None
MERC_PROJ4 = "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +no_defs +over"


def shapes_json(request):
    if request.GET.get('qs'):
        qs = eval(request.GET['qs'])
    else:
        qs = AlphaShapes.objects.all()[:200]
    return render_to_geojson(qs,exclude=['created'])
    
def slippy_map(request):
    return render_to_response('map.html', {})

class MapCache(object):
    def __init__(self,request):
        self.map = mapnik.Map(1,1)
        mapfile = request.GET['LAYERS'].split(',')[0] + '.xml'
        mapfile_path = os.path.join(settings.MAPFILES,mapfile)
        mapnik.load_map(self.map,str(mapfile_path))
        try:
            p = mapnik.Projection('+init=%s' % str(request.GET['SRS']) )
        except:
            p = mapnik.Projection(MERC_PROJ4)
        self.map.srs = p.params()

def mapnik_tiles(request):
    global MAP_CACHE
    w,h = int(request.GET['WIDTH']), int(request.GET['HEIGHT'])
    mime = request.GET['FORMAT']
    if not MAP_CACHE:
        MAP_CACHE = MapCache(request)
    env = map(float,request.GET['BBOX'].split(','))
    tile = MAP_CACHE.map
    tile.buffer_size = 128
    tile.resize(w,h)
    tile.zoom_to_box(mapnik.Envelope(*env))
    draw = mapnik.Image(tile.width, tile.height)
    mapnik.render(tile,draw)
    image = draw.tostring(str(mime.split('/')[1]))
    response = HttpResponse()
    response['Content-length'] = len(image)
    response['Content-Type'] = mime
    response.write(image)
    return response