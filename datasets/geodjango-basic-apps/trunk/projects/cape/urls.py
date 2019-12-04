from django.conf.urls.defaults import *
from lionshead.views import *
from settings import STATIC_FILES

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # Example:
    # (r'^cape/', include('cape.foo.urls')),

    # Uncomment the admin/doc line below and add 'django.contrib.admindocs' 
    # to INSTALLED_APPS to enable admin documentation:
    # (r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    (r'^admin/(.*)', admin.site.root),
    (r'^kml/', all_kml),
    (r'^$', map_page),
    (r'^wards/(?P<id>[0-9]*)/', ward),

    (r'^static/(?P<path>.*)$', 'django.views.static.serve', {'document_root': STATIC_FILES, 'show_indexes': True}),

)
