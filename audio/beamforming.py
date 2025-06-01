# This is testing how well a speaker array, where the delay is controlled for each speaker,can
# emit localized sound.
import numpy as np
import soundfile as sf
import sounddevice as sd
from matplotlib import pyplot as plt

# Speed of sound, in m/s
SPEED_OF_SOUND_M_S = 343.0

# Load audio data from file and write it as a numpy array
def load_audio_file(file_path):
    """Load an audio file and return the audio data and sample rate."""
    audio_data, sample_rate = sf.read(file_path)
    # If the audio is too high of a sample rate, downsample it
    while sample_rate > 24000:
        audio_data = audio_data[::2]
        sample_rate //= 2
    return audio_data, sample_rate

track1_file = "/Users/droz/Documents/GitHub/misc/audio/samples/position1_rep.wav"
track2_file = "/Users/droz/Documents/GitHub/misc/audio/samples/position2_rep.wav"
track1, sr1 = load_audio_file(track1_file)
track2, sr2 = load_audio_file(track2_file)

# Ensure both tracks have the same sample rate
if sr1 != sr2:
    raise ValueError("Sample rates of the two tracks do not match.")
# Pad the shorter track with zeros to match the length of the longer track
if len(track1) < len(track2):
    track1 = np.pad(track1, (0, len(track2) - len(track1)), mode='constant')
elif len(track2) < len(track1):
    track2 = np.pad(track2, (0, len(track1) - len(track2)), mode='constant')
# Loop the tracks a few times
track1 = np.tile(track1, 2)
track2 = np.tile(track2, 2)

# Use the longest of the two tracks to figure out the length of the output
tmax = len(track1) / sr1  # in seconds
t = np.linspace(0, tmax, num=len(track1))  # time vector
num_samples = t.shape[0]  # total number of samples in the output

# The location of the two points we are trying to project to
target1 = np.array([-1, 0, 10])  # meters
target2 = np.array([1, 0, 10])  # meters

# The position of the listener in the room
# For now we are just moving the listener between two points in space
point1 = np.array([-3, 0, 10])  # meters
point2 = np.array([3, 0, 10])  # meters
listener_pos = np.linspace(point1, point2, num=num_samples)

# This is the position of each speaker in the array. They are arranged on a regular grid.
num_speakers_x = 100
num_speakers_y = 1
speaker_array_size = 14  # meters
speaker_spacing = speaker_array_size / num_speakers_x  # meters
speakers_x = np.linspace(-num_speakers_x / 2 * speaker_spacing, num_speakers_x / 2 * speaker_spacing, num_speakers_x)
speakers_y = np.linspace(-num_speakers_y / 2 * speaker_spacing, num_speakers_y / 2 * speaker_spacing, num_speakers_y)
speakers_x, speakers_y = np.meshgrid(speakers_x, speakers_y)
speakers_x = speakers_x.flatten()
speakers_y = speakers_y.flatten()
speakers_z = np.zeros_like(speakers_x)  # Assuming all speakers are at z=0
speakers_pos = np.vstack((speakers_x, speakers_y, speakers_z)).T

def propagation_effects(speaker_pos, targets, speed_of_sound):
    """ Calculate the propagation delay in seconds for a speaker based on the distance to the target point
    Args:
        speaker_pos (np.ndarray): speaker position (3).
        target_pos (np.ndarray): Target positions, as a function of time (3, N).
        speed_of_sound (float): Speed of sound in m/s.
    Returns:
        np.ndarray: Delay in seconds
        np.ndarray: Attenuation factor based on distance.
    """
    if targets.ndim == 1:
        distances = np.linalg.norm(speaker_pos - targets, axis=0)
    else:
        distances = np.linalg.norm(speaker_pos - targets, axis=1)
    delays = distances / speed_of_sound
    attenuations = 1 / distances**2  # Simple attenuation model
    return delays, attenuations

# For each speaker, we compute the waveform at our current position in time
audio = np.zeros(num_samples)

# The speaker array has a finite size and a very steep dropoff. IF we don't do anything
# this will result into very pronounced side lobes. To mitigate that, we apply a windowing function
# (basically we smoothly reduce the amplitude of the speakers at the edges of the array).
# We use a modified hanning window (with a flat middle section).
num_speakers = len(speakers_pos)
window = np.hanning(num_speakers//2)
window = np.concatenate((window[:num_speakers//4], np.ones(num_speakers//2), window[num_speakers//4:]))

for i, (speaker_pos, window_val) in enumerate(zip(speakers_pos, window)):
    print(f"Processing speaker ({i}/{num_speakers}) at position: {speaker_pos}")
    beamforming_delay1, beamforming_attenuation1 = propagation_effects(speaker_pos, target1, SPEED_OF_SOUND_M_S)
    beamforming_delay2, beamforming_attenuation2 = propagation_effects(speaker_pos, target2, SPEED_OF_SOUND_M_S)

    # compute the delay for the listener position
    listener_delays, attenuations = propagation_effects(speaker_pos, listener_pos, SPEED_OF_SOUND_M_S)
    # Sample the audio track at the appropriate time for each speaker
    delays1 = - beamforming_delay1 + listener_delays
    delays2 = - beamforming_delay2 + listener_delays
    audio += window_val * attenuations * np.interp(t + delays1, t, track1 / beamforming_attenuation1, left=0, right=0)
    audio += window_val * attenuations * np.interp(t + delays2, t, track2 / beamforming_attenuation2, left=0, right=0)

audio /= sum(window)  # Normalize by the windowing function

# Playback the audio tracks
sd.play(audio, samplerate=sr1)
sd.wait()  # Wait until the audio is finished playing