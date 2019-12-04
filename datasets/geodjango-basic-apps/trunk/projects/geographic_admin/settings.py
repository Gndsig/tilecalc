# Django global settings

# This allows us to construct the needed absolute paths dynamically,
# e.g., for the GIS_DATA_DIR, MEDIA_ROOT, and TEMPLATE_DIRS settings.
# see: http://rob.cogit8.org/blog/2008/Jun/20/django-and-relativity/
import os
GEOGRAPHIC_ADMIN_DIR = os.path.dirname(__file__)

# http://docs.djangoproject.com/en/dev/topics/testing/#id1
# Your user must be a postgrest superuser
# Avoid specifying your password with: ~/.pgpass
# http://www.postgresql.org/docs/8.3/interactive/libpq-pgpass.html
TEST_RUNNER='django.contrib.gis.tests.run_gis_tests'

DEBUG=True

ADMINS = (
    # ('Your Name', 'your_email@domain.com'),
)

DATABASE_ENGINE = 'postgresql_psycopg2'
DATABASE_NAME = 'geoadmin'
DATABASE_USER = 'postgres' # make sure to change this to your postgres user
DATABASE_PASSWORD = ''
DATABASE_HOST = ''
DATABASE_PORT = ''


# django 1.2 compatible database settings
DATABASES = {
    'default': {
        'NAME': DATABASE_NAME, # add in local settings
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'USER': DATABASE_USER,
        'PASSWORD': DATABASE_PASSWORD
    }
}

# Not used at this point but you'll need it here if you 
# want to enable a google maps baselayer within your
# OpenLayers maps
GOOGLE_MAPS_API_KEY='abcdefg'

GIS_DATA_DIR = os.path.join(GEOGRAPHIC_ADMIN_DIR, 'data')

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'America/Vancouver'

SITE_ID = 1

USE_I18N = True

MEDIA_ROOT = os.path.join(GEOGRAPHIC_ADMIN_DIR, 'media')

MEDIA_URL = '/media/'

ADMIN_MEDIA_PREFIX = '/admin_media/'

SECRET_KEY = '2f!vq4!f)u#g-sk7_=z+i0e(o0o&hue5khxbdkdx$f%hvpb^vd'

TEMPLATE_LOADERS = (
    'django.template.loaders.filesystem.Loader',
    'django.template.loaders.app_directories.Loader',
)

from django.conf.global_settings import TEMPLATE_CONTEXT_PROCESSORS
TEMPLATE_CONTEXT_PROCESSORS += (
     'django.core.context_processors.request',
) 

MIDDLEWARE_CLASSES = (
    'django.middleware.common.CommonMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.middleware.doc.XViewMiddleware',
)

ROOT_URLCONF = 'urls'

TEMPLATE_DIRS = (
    os.path.join(GEOGRAPHIC_ADMIN_DIR, 'templates'),
    # Don't forget to use absolute paths, not relative paths.
)

INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.sites',
    'django.contrib.databrowse',
    'django.contrib.gis',
    'world',
)
