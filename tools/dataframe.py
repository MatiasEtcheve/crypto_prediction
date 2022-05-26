import re
from datetime import timedelta


def convert_to_timedelta(interval: str, ago: int) -> timedelta:
    """Convert a tuple of interval, and number into a timedelta.
    For instance, if `interval="2d"`, and `ago=300`, it will return `timedelta(days=2*300)`.

    Args:
        interval (str): interval in terms of binance constant
        ago (int): number of interval in timedelta

    Returns:
        timedelta: the corresponding timedelta instance
    """
    time_equivalence = {"d": "days", "h": "hours", "m": "minutes"}
    interval_value, interval_base = re.findall("\d+|\D+", interval)
    interval_value = ago * int(interval_value)
    interval_base = time_equivalence[interval_base]
    return timedelta(**{interval_base: interval_value})
