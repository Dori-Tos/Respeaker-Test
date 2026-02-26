from custom_tuning import Tuning
import usb.core
import time
import pyaudio
import numpy as np
import math


def find_respeaker_device():
    """Find the Respeaker device index in PyAudio"""
    p = pyaudio.PyAudio()
    respeaker_index = None
    
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        max_input = info.get('maxInputChannels', 0)
        name = info.get('name', '')
        
        if isinstance(max_input, int) and max_input > 0:
            print(f"  {i}: {name} (channels: {max_input})")
            # Look for ReSpeaker in the device name
            if isinstance(name, str) and ('respeaker' in name.lower() or 'seeed' in name.lower()):
                respeaker_index = i
                print(f"    -> Found ReSpeaker device!")
    
    p.terminate()
    return respeaker_index


def get_gain(device_index=None):
    # Find the Respeaker 4 mic array USB device
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("Respeaker 4 mic array not found")
        return None
    
    # Audio parameters
    RESPEAKER_RATE = 16000
    RESPEAKER_CHANNELS = 1
    RESPEAKER_WIDTH = 2
    CHUNK = 1024
    SAMPLE_DURATION = 0.5  # seconds to sample for level measurement
    
    # Find ReSpeaker device if not specified
    if device_index is None:
        print("\nSearching for ReSpeaker device...")
        device_index = find_respeaker_device()
        if device_index is None:
            print("\nWarning: Could not auto-detect ReSpeaker. Using default index 1.")
            print("If this fails, manually specify the correct device index.")
            device_index = 1
        else:
            print(f"\nUsing device index: {device_index}")
    
    RESPEAKER_INDEX = device_index
    
    # Create Tuning object to interact with the device
    tuning = Tuning(mic_array)
    
    try:
        # Get current AGC gain
        agc_gain = tuning.read('AGCGAIN')
        agc_gain_db = 20 * math.log10(agc_gain) if agc_gain and agc_gain > 0 else -float('inf')
        
        # Get AGC max gain and desired level
        agc_max_gain = tuning.read('AGCMAXGAIN')
        agc_desired_level = tuning.read('AGCDESIREDLEVEL')
        
        # Get DOA (Direction of Arrival) angle
        doa_angle = tuning.direction
        
        # Get voice activity status
        voice_activity = tuning.is_voice
        
        print(f"AGC Gain (applied): {agc_gain:.2f} (dB: {agc_gain_db:.2f})")
        print(f"AGC Max Gain: {agc_max_gain:.2f}")
        print(f"AGC Desired Level: {agc_desired_level:.6f}")
        print(f"DOA Angle: {doa_angle}°")
        print(f"Voice Activity: {'Yes' if voice_activity else 'No'}")
        
        # Now measure the actual audio signal level
        print(f"\nSampling audio for {SAMPLE_DURATION}s to measure real signal level...")
        
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                rate=RESPEAKER_RATE,
                format=p.get_format_from_width(RESPEAKER_WIDTH),
                channels=RESPEAKER_CHANNELS,
                input=True,
                input_device_index=RESPEAKER_INDEX,
            )
            
            frames = []
            num_chunks = int(RESPEAKER_RATE / CHUNK * SAMPLE_DURATION)
            
            for i in range(num_chunks):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            
            # Calculate RMS (Root Mean Square) level
            rms = np.sqrt(np.mean(audio_data**2))
            
            # Calculate peak level
            peak = np.max(np.abs(audio_data))
            
            # Convert to dBFS (dB relative to full scale)
            # For 16-bit audio, full scale is 32768
            full_scale = 32768.0
            rms_dbfs = 20 * math.log10(rms / full_scale) if rms > 0 else -float('inf')
            peak_dbfs = 20 * math.log10(peak / full_scale) if peak > 0 else -float('inf')
            
            print(f"\nReal Audio Signal Levels:")
            print(f"  RMS Level: {rms:.2f} ({rms_dbfs:.2f} dBFS)")
            print(f"  Peak Level: {peak:.2f} ({peak_dbfs:.2f} dBFS)")
            
            return {
                'agc_gain': agc_gain,
                'agc_gain_db': agc_gain_db,
                'agc_max_gain': agc_max_gain,
                'agc_desired_level': agc_desired_level,
                'doa_angle': doa_angle,
                'voice_activity': voice_activity,
                'rms_level': rms,
                'rms_dbfs': rms_dbfs,
                'peak_level': peak,
                'peak_dbfs': peak_dbfs
            }
        except Exception as e:
            print(f"\nError sampling audio: {e}")
            print("Returning device parameters only (without audio level measurement)")
            return {
                'agc_gain': agc_gain,
                'agc_gain_db': agc_gain_db,
                'agc_max_gain': agc_max_gain,
                'agc_desired_level': agc_desired_level,
                'doa_angle': doa_angle,
                'voice_activity': voice_activity,
                'error': str(e)
            }
    finally:
        # Close the device connection
        tuning.close()


if __name__ == '__main__':
    import sys
    print("Reading gain and DOA from Respeaker 4 mic array...")
    print("-" * 50)
    
    # Allow specifying device index as command line argument
    device_idx = None
    if len(sys.argv) > 1:
        try:
            device_idx = int(sys.argv[1])
            print(f"Using specified device index: {device_idx}")
        except ValueError:
            print(f"Invalid device index: {sys.argv[1]}")
            print("Usage: python get_gain.py [device_index]")
            sys.exit(1)
    
    get_gain(device_idx)
