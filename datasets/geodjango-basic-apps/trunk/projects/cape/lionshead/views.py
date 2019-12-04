from django.shortcuts import render_to_response, get_object_or_404
from django.contrib.gis.shortcuts import render_to_kml
from lionshead.models import *

def all_kml(request):
    locations  = InterestingLocation.objects.kml()
    return render_to_kml("gis/kml/placemarks.kml", {'places' : locations})


def map_page(request):
    lcount = InterestingLocation.objects.all().count()
    return render_to_response('map.html', {'location_count' : lcount})

def ward(request, id):
    ward = get_object_or_404(Ward, pk=id)
    interesting_points = InterestingLocation.objects.filter(geometry__intersects=ward.geometry)
    return render_to_response("ward.html", { 'ward': ward, 'interesting_points': interesting_points })
