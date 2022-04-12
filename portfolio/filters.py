class FilterException(Exception):
    def __init__(self, message="", **kwargs):
        self.__dict__.update(kwargs)
        Exception.__init__(self, message)


class PriceFilterException(FilterException):
    pass


class HighPriceFilterException(PriceFilterException):
    pass


class LowPriceFilterException(PriceFilterException):
    pass


class PercentPriceException(FilterException):
    pass


class HighPercentPriceException(PercentPriceException):
    pass


class LowPercentPriceException(PercentPriceException):
    pass


class LotSizeException(FilterException):
    pass


class LowSizeException(LotSizeException):
    pass


class HighSizeException(LotSizeException):
    pass


class MinNotionalException(LotSizeException):
    pass
