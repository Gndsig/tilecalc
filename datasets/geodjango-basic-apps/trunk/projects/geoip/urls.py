from django.conf import settings
from django.conf.urls.defaults import *
from django.contrib import admin

from whereami.views import welcome, whereami, whereis
admin.autodiscover()

urlpatterns = patterns('',
    (r'^$', welcome),
    (r'^whereami/', whereami),
    (r'^whereis/(.*)', whereis),
)
