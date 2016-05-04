
from .load_data import load_data
from .DeepSea import DeepSea
from .MotifSimulation_binary import MotifSimulation_binary
from .MotifSimulation_categorical import MotifSimulation_categorical
from .MotifSimulation_correlated import MotifSimulation_correlated
from .Basset import Basset

__all__ = [
		   'load_data',
		   'Basset',
		   'DeepSea',
           'MotifSimulation_binary',
           'MotifSimulation_categorical',
           'MotifSimulation_correlated'
           ]
