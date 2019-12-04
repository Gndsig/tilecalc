from django.contrib.gis.db import models

class InterestingLocation(models.Model):
    """A spatial model for interesting locations """
    name = models.CharField(max_length=50)
    description = models.TextField()
    interestingness = models.IntegerField()
    geometry = models.PointField(srid=4326) 
    objects = models.GeoManager() # so we can use spatial queryset methods

    def __unicode__(self): return self.name

class Ward(models.Model):
    """Spatial model for Cape Town Wards"""
    subcouncil = models.CharField(max_length=20)
    party = models.CharField(max_length=50)
    ward = models.CharField(max_length=50)
    cllr = models.CharField(max_length=150)
    geometry = models.MultiPolygonField(srid=4326)
    objects = models.GeoManager()

    def __unicode__(self): return self.cllr
