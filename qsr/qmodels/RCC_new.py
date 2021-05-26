"""
Methods are adapted from https://github.com/strands-project/ \
strands_qsr_lib/blob/master/qsr_lib/src/qsrlib_qsrs/qsr_rcc*.py
"""
import time
import numpy as np
from shapely.geometry import Polygon  # 多边形

def compute_rcc_rel(mbr1, mbr2, m=8):
    rcc8_rel = _compute_rcc8_rel(mbr1, mbr2)
    # relation mapping
    if m == 5:
        return _rcc5_rel_mapper(rcc8_rel)
    elif m == 4:
        return _rcc4_rel_mapper(rcc8_rel)
    elif m == 3:
        return _rcc3_rel_mapper(rcc8_rel)
    elif m == 2:
        return _rcc2_rel_mapper(rcc8_rel)
    else:
        return rcc8_rel

def areaCalPoly(poly):
    polygon = Polygon(poly).convex_hull

    return polygon.area

def overlapAreaPloys(poly1, poly2):
    poly1 = Polygon(poly1).convex_hull  # Polygon：多边形对象
    poly2 = Polygon(poly2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area

def isVertexOnSegment(poi,s_poi,e_poi):
    """Check each vertex in polygon 1 on the edges of polygon 2
    :param poi: Coordinate [X, Y] of the vertex(m) in polygon 1
    :param s_poi: Coordinate [X, Y] of the start vertex(n) in polygon 2
    :param e_poi: Coordinate [X, Y] of the end vertex(n+1: clockwise) in polygon 2
    :return: Boolean value of m is on the line segment (n, n+1)
    """
    if (poi[0]-s_poi[0])*(e_poi[1]-s_poi[1]) == (e_poi[0]-s_poi[0])*(poi[1]-s_poi[1]) and \
                            min(s_poi[0], e_poi[0])<=poi[0]<=max(s_poi[0], e_poi[0]) and \
                            min(s_poi[1], e_poi[1])<=poi[1]<=max(s_poi[1], e_poi[1]):
        return True
    else:
        return False

def isTanWithPoly(poi,poly):
    """Check each vertex in polygon 1 is in polygon 2
    :param poi: Coordinate [X, Y] of the vertex(m) in polygon 1
    :param poly: List of coordinate [X, Y] of the vertex in polygon 2
    :return: Boolean value of vertex(m) is in polygon 2
    """
    stan = 0
    for i in range(len(poly)-1):
        s_poi=poly[i]
        e_poi=poly[i+1]
        if isVertexOnSegment(poi,s_poi,e_poi):
            stan+=1

    return stan != 0

def _compute_rcc8_rel(poly1, poly2):
    """Calculate the location relation of two polygons
    :param poly1: 3-D coordinate vectors of polygon 1
    :param poly2: 3-D coordinate vectors of polygon 2
    :return: RCC result of the two polygons
    """

    pinsc = 0
    ptan = 0
    pinsci = 0
    ptani = 0

    for i in range(len(poly1)):
        if isTanWithPoly(poly1[i], poly2):
            ptan+=1

    for i in range(len(poly2)):
        if isTanWithPoly(poly2[i], poly1):
            ptani+=1
    # print(pinsc, ptan, pinsci, ptani)
    overlap_area = overlapAreaPloys(poly1, poly2)
    area_poly1 = areaCalPoly(poly1)
    area_poly2 = areaCalPoly(poly2)
    # print(overlap_area, ptan, ptani, area_poly1, area_poly2)
    if overlap_area==0:
        if ptan+ptani ==0:
            return "dc"
        else:
            return "ec"
    else:
        if overlap_area < area_poly1 and overlap_area< area_poly2:
            return "po"
        if overlap_area == area_poly1 == area_poly2:
            return "eq"
        if overlap_area == area_poly1 and overlap_area < area_poly2:
            if ptan + ptani ==0:
                return "ntpp"
            else:
                return "tpp"
        if overlap_area == area_poly2 and overlap_area < area_poly1:
            if ptan + ptani == 0:
                return "ntppi"
            else:
                return "tppi"



def _rcc5_rel_mapper(rcc8_rel):
    switcher = {
        "dc": "dc",
        "ec": "dc",
        "po": "po",
        "tpp": "pp",
        "ntpp": "pp",
        "tppi": "ppi",
        "ntppi": "ppi",
        "eq": "eq"
    }
    return switcher.get(rcc8_rel)


def _rcc4_rel_mapper(rcc8_rel):
    switcher = {
        "dc": "dc",
        "ec": "po",
        "po": "po",
        "tpp": "pp",
        "ntpp": "pp",
        "eq": "pp",
        "tppi": "ppi",
        "ntppi": "ppi"
    }
    return switcher.get(rcc8_rel)


def _rcc3_rel_mapper(rcc8_rel):
    switcher = {
        "dc": "dc",
        "ec": "po",
        "po": "po"
    }
    return switcher.get(rcc8_rel, "in")


def _rcc2_rel_mapper(rcc8_rel):
    switcher = {
        "dc": "dc"
    }
    return switcher.get(rcc8_rel, "c")


def get_rcc_rels(m=8):
    if m == 2:
        return ['dc', 'c']
    if m == 3:
        return ['dc', 'po', 'in']
    if m == 4:
        return ['dc', 'po', 'pp', 'ppi']
    if m == 5:
        return ['dc', 'po', 'pp', 'ppi', 'eq']
    return ['dc', 'ec', 'po', 'tpp', 'tppi', 'ntpp', 'ntppi', 'eq']


if __name__ == '__main__':
    m1 = [[-148.0, -96.0], [148.0, 96.0], [148.0, 96.0], [148.0, 96.0], [145.0, 96.0], [145.0, 96.0], [142.0, 91.0],
     [142.0, 91.0], [140.0, 91.0], [140.0, 91.0], [141.0, 87.0], [141.0, 87.0], [143.0, 84.0], [143.0, 84.0],
     [151.0, 84.0], [151.0, 84.0], [152.0, 86.0], [152.0, 86.0], [154.0, 91.0], [154.0, 91.0], [152.0, 92.0],
     [152.0, 92.0]]
    m2 = [[566.0, 76.0], [566.0, 76.0], [559.0, 76.0], [559.0, 76.0], [559.0, 103.0], [559.0, 103.0], [566.0, 103.0],
     [566.0, 103.0]]


    print(compute_rcc_rel(m1, m2))
