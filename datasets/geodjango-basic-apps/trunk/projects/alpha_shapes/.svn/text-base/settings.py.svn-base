
import os
PROJECT_DIR = os.path.dirname(__file__)

DEBUG=False

ADMINS = (
    # ('Your Name', 'your_email@domain.com'),
)

DATABASE_ENGINE = 'postgresql_psycopg2'
DATABASE_NAME = 'clustr'
DATABASE_USER = 'postgres'
DATABASE_PASSWORD = ''
DATABASE_HOST = ''
DATABASE_PORT = ''

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'America/Vancouver'

SITE_ID = 1


MEDIA_ROOT = os.path.join(PROJECT_DIR, 'media')

MEDIA_URL = '/media/'

# for mapnik xml file for the WMS layer in the map
MAPFILES = os.path.join(PROJECT_DIR, 'mapfiles')

ADMIN_MEDIA_PREFIX = '/admin_media/'

SECRET_KEY = '2f!vq4!f)u#g-sk7_=z+i0e(o0o&hue5khxbdkdx$f%hvpb^vd'

TEMPLATE_LOADERS = (
    'django.template.loaders.filesystem.load_template_source',
    'django.template.loaders.app_directories.load_template_source',
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
    os.path.join(PROJECT_DIR, 'templates'),
)

INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.sites',
    'django.contrib.databrowse',
    'django.contrib.gis',
    'clustr',
)
