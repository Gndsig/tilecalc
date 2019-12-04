# -*- coding: utf-8 -*-
# models.py

from django.contrib.gis.db import models

class AlphaShapes(models.Model):
    woe_id = models.IntegerField() 
    place_id = models.CharField(max_length=255)
    place_type = models.CharField(max_length=255)
    place_type_id = models.IntegerField()
    label = models.TextField()
    created = models.DateTimeField()
    poly = models.PolygonField(srid=4326)
    point = models.PointField(srid=4326)
    area = models.FloatField()
    objects = models.GeoManager()

    class Meta:
        verbose_name_plural = "Alpha Shapes"

    # Returns the string representation of the model.        
    def __unicode__(self):
        return unicode('%s: %s' % (self.woe_id,self.label))
    
    def name(self):
        return self.__unicode__()

    def save(self, force_insert=False, force_update=False):
        # pre-calculate these just for now...
        if not self.area:
            self.area = self.poly.area
        if not self.point:
            self.point = self.poly.centroid
        
        super(AlphaShapes, self).save()