class TileSegment:
    def __init__(self):
        # type judgment
        
        pass

class LinkingPolylineTile:
    
    # constructor for WKT
    def __init__(self, wkt_polyline, TileSegment):
        self.wkt_polyline = wkt_polyline
        self.TileSegment = TileSegment
    
    # Calculation Minimum Bounding Rectangle aligned xy axis.
    #@classmethod
    def xy_aligned(self, minimum=None):
        
        pass
    
    def overlappingTiles(self, zoom):
        # XY coordinate of XYZTile covering WKTPolyline
        self.wkt_polyline
        self.TileSegment
        pass
    
    def terminal_node_aligned(self, minimum=None):
        
        pass
    
    def overlappingTileSegments(self, zoom, mbr_type):
        pass