from django.conf.urls import patterns, url
from django.conf import settings
#import django.views.static

from web.views import connections

urlpatterns = patterns('',
    url(r'^$', 'web.views.home', name='home'),
    url(r'^(?:chart/)?(?:[Cc]onnect(?:ion)?s?|[Gg]raph)/(?P<edges>[^/]*)', connections),
    url(r'^static/(?P<path>.*)$', 'django.views.static.serve', { 'document_root': settings.STATIC_ROOT} ),
    url(r'^media/(?P<path>.*)$', 'django.views.static.serve', { 'document_root': settings.MEDIA_ROOT} ),
)
