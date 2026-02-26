from .Microphone import Microphone
from .DOAEstimator import DOAEstimator
from .Beamformer import Beamformer
from .EchoCanceller import EchoCanceller
from .Filter import Filter
from .AGC import AGC
from .Codec import Codec


class Array:
    def __init__(self, id_vendor, id_product, mic_list: list[Microphone], doa_estimator: DOAEstimator, beamformer: Beamformer, 
                 echo_canceller: EchoCanceller, filters: list[Filter], agc: AGC, codec: Codec):
        
        self.id_vendor: int = id_vendor
        self.id_product: int = id_product
        self.mic_list: list[Microphone] = mic_list
        self.doa_estimator: DOAEstimator = doa_estimator
        self.beamformer: Beamformer = beamformer
        self.echo_canceller: EchoCanceller = echo_canceller
        self.filters: list[Filter] = filters
        self.agc: AGC = agc
        self.codec: Codec = codec
        
        
        self.beamformer.set_microphones(mic_list)
        
    
    