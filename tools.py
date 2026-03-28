import math


def calculate_distance(lat1, lon1, lat2, lon2) -> float:
    """Calculates the distance between two locations.

    Args:
        lat1 (float): the latitude of the first point
        lon1 (float): the longtiude of the first point
        lat2 (float): the latitude of the second point
        lon2 (float): the longtiude of the second point

    Returns:
        float: the distance between the two points
    """
    R = 6371

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat1 - lat2
    dlon = lon1 - lon2

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    return R * c
