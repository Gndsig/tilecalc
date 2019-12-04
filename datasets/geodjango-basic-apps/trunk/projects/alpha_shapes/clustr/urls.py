from django.conf import settings
from django.conf.urls.defaults import *

from clustr.views import *

urlpatterns = patterns('',
    (r'^$', slippy_map),
    (r'^shapes/$', shapes_json),
    (r'^tiles/$', mapnik_tiles),
    )