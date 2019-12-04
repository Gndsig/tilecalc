# python modules
import re
import urllib2 

# django imports
from django.conf import settings
from django.shortcuts import render_to_response, get_list_or_404

# geodjango import
from django.contrib.gis.utils import GeoIP

# Django imports to contruct welcome message
from django.template import Template, Context
from django.http import HttpResponse

def get_my_ip():
    """
    Fetch an IP when request.META returns localhost
    """
    
    checkip = urllib2.urlopen('http://checkip.dyndns.org/').read() 
    matcher = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}') 
    try:
        remote_ip = matcher.search(checkip).group()
    except:
        remote_ip = ''
    return remote_ip

def whereami(request):
	g = GeoIP()
	remote_ip = request.META['REMOTE_ADDR']
	if remote_ip == '127.0.0.1':
	    remote_ip = get_my_ip()
	remote_location = g.city(remote_ip)
	if remote_location:
	    return render_to_response('gmaps.html', {'remote_location': remote_location, 'GOOGLE_MAPS_API_KEY':settings.GOOGLE_MAPS_API_KEY})
	else: # localhost ip cannot be found
	    return render_to_response('not_found.html')
	    
def whereis(request, ip):
	g = GeoIP()
	ip_match = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
	try:
	    remote_ip = ip_match.findall(ip)[0]
	except:
	    remote_ip = ''
	remote_location = g.city(remote_ip)
	if remote_location:
	    return render_to_response('gmaps.html', {'remote_location': remote_location, 'GOOGLE_MAPS_API_KEY':settings.GOOGLE_MAPS_API_KEY})
	else: # localhost ip cannot be found
	    return render_to_response('not_found.html')

def welcome(request):
    """
    Create a Welcome Page for GeoDjango Sample Project with Template as a string
    """
    t = Template(WELCOME, name='Geodjango Welcome')
    c = Context( {'project_name': settings.SETTINGS_MODULE.split('.')[0], 'my_ip': get_my_ip()} )
    return HttpResponse(t.render(c), mimetype='text/html')

WELCOME = """
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html lang="en"><head>
  <meta http-equiv="content-type" content="text/html; charset=utf-8">
  <meta name="robots" content="NONE,NOARCHIVE"><title>Welcome to GeoDjango</title>
  <style type="text/css">
    html * { padding:0; margin:0; }
    body * { padding:10px 20px; }
    body * * { padding:0; }
    body { font:small sans-serif; }
    body>div { border-bottom:1px solid #ddd; }
    h1 { font-weight:normal; }
    h2 { margin-bottom:.8em; }
    h2 span { font-size:80%; color:#666; font-weight:normal; }
    h3 { margin:1em 0 .5em 0; }
    h4 { margin:0 0 .5em 0; font-weight: normal; }
    table { border:1px solid #ccc; border-collapse: collapse; width:100%; background:white; }
    tbody td, tbody th { vertical-align:top; padding:2px 3px; }
    thead th { padding:1px 6px 1px 3px; background:#fefefe; text-align:left; font-weight:normal; font-size:11px; border:1px solid #ddd; }
    tbody th { width:12em; text-align:right; color:#666; padding-right:.5em; }
    ul { margin-left: 2em; margin-top: 1em; }
    #summary { background: #e0ebff; }
    #summary h2 { font-weight: normal; color: #666; }
    #explanation { background:#eee; }
    #instructions { background:#f6f6f6; }
    #summary table { border:none; background:transparent; }
  </style>
</head>

<body>
<div id="summary">
  <h1>Welcome!</h1>
  <h2>Congratulations on your GeoDjango-powered project: <u>{{project_name|capfirst}}</u></h2>
</div>

<div id="instructions">
  <p><b>{{project_name|capfirst}}</b> allows for the quick and dirty geocoding of remote/external IP addresses</p>
  <ul>
    <li>Visit <b><a href="whereami/">/whereami</a></b> to see your geocoded location (if you are running localhost your external IP will be fetched).</li>
    <li>The <b>/whereis</b> view allows you to manually enter various external IP addresses to test where they are. Try yours first: <a href="/whereis/{{ my_ip }}/">/whereis/{{ my_ip }}/</a></li>
  </ul>
</div>

<div id="explanation">Have fun!
</div>
</body></html>
"""
