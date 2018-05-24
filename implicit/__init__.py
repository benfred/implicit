from .als import alternating_least_squares

from . import als
from . import bpr
from . import nearest_neighbours

__version__ = '0.3.6'

__all__ = [alternating_least_squares, als, bpr, nearest_neighbours, __version__]
