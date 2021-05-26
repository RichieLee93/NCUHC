from __future__ import print_function, division

from math import radians, cos, sin, degrees, atan2, sqrt, asin, pi, floor
from shapely.geometry import Polygon


def  directionDistanceRelations(poly1, poly2):
    """Calculate direction and distance relation of two polygons
    :param poly1: coordinate of polygon1 [[x1, y1], [x2, y2], ....[xn, yn]]
    :param poly2: coordinate of polygon2 [[x1, y1], [x2, y2], ....[xn, yn]]
    :return: derection relation index
    """
    # center1, center2 = center_geolocation(poly1), center_geolocation(poly2)
    center1, center2 = list(Polygon(poly1).centroid.coords)[0], list(Polygon(poly2).centroid.coords)[0]
    Dist = sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    # Finds the differnece between the centres of each polygon
    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]

    if dx == 0 and dy == 0:
        direction = 'eq'
    else:
        # Calculate the angle of the line between the two objects (in degrees)
        angle = (atan2(dx,dy) * (180/pi))+22.5
        # If that angle is negative, invert it
        if angle < 0.0:
            angle = (360.0 + angle)
        angle_direct = {0: [7, 8], 1: [9, 10], 2: [11, 12], 3: [13, 14], 4: [0, 15, 16], 5: [1, 2], 6: [3, 4], 7: [5, 6]}
        for k, v in angle_direct.items():
            if floor((angle / 22.5)) in v:
                direction = __direction_switch(k)

    # Lookup labels and return answer
    return direction, round(Dist, 2), [round(center1[0], 1), round(center1[1], 1), round(center2[0], 1), round(center2[1], 1)]

def __direction_switch(x):
    """Switch Statement convert number into region label
    :param derection relation index
    :return: QSR relation.
    """
    return {
        0: 's',
        1: 'sw',
        2: 'w',
        3: 'nw',
        4: 'n',
        5: 'ne',
        6: 'e',
        7: 'se',
    }.get(x)


# def center_geolocation(geolocations):
#     """Calculate center point of polygon
#     :param Coordinate vector of [[x1, y1], [x2, y2], ....[xn, yn]] of the polygon
#     :return: center coordinate [xc, yc]
#     """
#     x = 0
#     y = 0
#     z = 0
#     lenth = len(geolocations)
#     for lon, lat in geolocations:
#         lon = radians(float(lon))
#         #  radians(float(lon))   Convert angle x from degrees to radians
#         lat = radians(float(lat))
#         x += cos(lat) * cos(lon)
#         y += cos(lat) * sin(lon)
#         z += sin(lat)
#         x = float(x / lenth)
#         y = float(y / lenth)
#         z = float(z / lenth)
#     return [degrees(atan2(y, x)), degrees(atan2(z, sqrt(x*x + y*y)))]


if __name__ == '__main__':
    m1 = [[354.27288818359377, 105.6636962890625], [354.2618103027344, 111.82369995117188],
          [411.94171142578127, 111.92742156982422], [411.9527893066406, 105.76744079589844]]
    m2 = [[266.3631286621094,130.3728485107422],[268.3177795410156,136.2145233154297],
        [323.01702880859377,117.9122543334961],[321.0624084472656,112.07058715820313]]
    print(directionDistanceRelations(m1, m2))
    m1 = [[354.27288818359377,105.6636962890625],[354.2618103027344,111.82369995117188],
          [411.94171142578127,111.92742156982422],[411.9527893066406,105.76744079589844]]
    m2 = [[264.56634521484377,128.403076171875],[266.1690673828125,134.35092163085938],
          [321.862548828125,119.34353637695313],[320.2597961425781,113.39569854736328]]

    print(directionDistanceRelations(m1, m2))
