from custom_tuning import Tuning
import usb.core
import time


def get_gain():
    mic_arrray = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    RESPEAKER_RATE = 16000
    RESPEAKER_CHANNELS = 1 # change base on firmwares, default_firmware.bin as 1 or i6_firmware.bin as 6
    RESPEAKER_WIDTH = 2
    # run getDeviceInfo.py to get index
    RESPEAKER_INDEX = 1  # refer to input device id
    CHUNK = 1024
    RECORD_SECONDS = 5
