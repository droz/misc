# This is testing how well a speaker array, where the delay is controlled for each speaker,can
# emit localized sound.
import numpy as np
import soundfile as sf
import sounddevice as sd

# Speed of sound, in m/s
SPEED_OF_SOUND_M_S = 343.0

# Load audio data from file and write it as a numpy array
track1_file = "/Users/droz/Documents/GitHub/misc/audio/samples/subject1.wav"
track2_file = "/Users/droz/Documents/GitHub/misc/audio/samples/subject2.wav"

track1, sr1 = sf.read(track1_file)
track2, sr2 = sf.read(track2_file)
# Ensure both tracks have the same sample rate
if sr1 != sr2:
    raise ValueError("Sample rates of the two tracks do not match.")
# Pad the shorter track with zeros to match the length of the longer track
if len(track1) < len(track2):
    track1 = np.pad(track1, (0, len(track2) - len(track1)), mode='constant')
elif len(track2) < len(track1):
    track2 = np.pad(track2, (0, len(track1) - len(track2)), mode='constant')
# Loop the tracks a few times
track1 = np.tile(track1, 5)
track2 = np.tile(track2, 5)

# Use the longest of the two tracks to figure out the length of the output
tmax = len(track1) / sr1  # in seconds
t = np.linspace(0, tmax, num=len(track1))  # time vector
num_samples = t.shape[0]  # total number of samples in the output

# The location of the two points we are trying to project to
target1 = np.array([0, 0, 5])  # meters
target2 = np.array([2, 0, 5])  # meters

# The position of the listener in the room
# For now we are just moving the listener between two points in space
point1 = np.array([-3, 0, 5])  # meters
point2 = np.array([3, 0, 5])  # meters
listener_pos = np.linspace(point1, point2, num=num_samples)

# This is the position of each speaker in the array. They are arranged on a regular grid.
num_speakers_x = 100
num_speakers_y = 1
speaker_spacing = 0.1  # meters
speakers_x = np.linspace(-num_speakers_x / 2 * speaker_spacing, num_speakers_x / 2 * speaker_spacing, num_speakers_x)
speakers_y = np.linspace(-num_speakers_y / 2 * speaker_spacing, num_speakers_y / 2 * speaker_spacing, num_speakers_y)
speakers_x, speakers_y = np.meshgrid(speakers_x, speakers_y)
speakers_x = speakers_x.flatten()
speakers_y = speakers_y.flatten()
speakers_z = np.zeros_like(speakers_x)  # Assuming all speakers are at z=0
speakers_pos = np.vstack((speakers_x, speakers_y, speakers_z)).T

def propagation_delays(speaker_pos, targets, speed_of_sound):
    """ Calculate the propagation delay in seconds for a speaker based on the distance to the target point
    Args:
        speaker_pos (np.ndarray): speaker position (3).
        target_pos (np.ndarray): Target positions, as a function of time (3, N).
        speed_of_sound (float): Speed of sound in m/s.
    """
    if targets.ndim == 1:
        distances = np.linalg.norm(speaker_pos - targets, axis=0)
    else:
        distances = np.linalg.norm(speaker_pos - targets, axis=1)
    delays = distances / speed_of_sound
    return delays

# For each speaker, we compute the waveform at our current position in time
audio = np.zeros(num_samples)
for speaker_pos in speakers_pos:
    beamforming_delay1 = - propagation_delays(speaker_pos, target1, SPEED_OF_SOUND_M_S)
    beamforming_delay2 = - propagation_delays(speaker_pos, target2, SPEED_OF_SOUND_M_S)

    # compute the delay for the listener position
    listener_delays = propagation_delays(speaker_pos, listener_pos, SPEED_OF_SOUND_M_S)
    # Sample the audio track at the appropriate time for each speaker
    delays1 = beamforming_delay1 + listener_delays
    delays2 = beamforming_delay2 + listener_delays
    audio += np.interp(t, t - delays1, track1)
    #audio += np.interp(t, t - delays2, track2, left=0, right=0)

audio /= len(speakers_pos)  # Normalize by the number of speakers

# Playback the audio tracks
sd.play(audio, samplerate=sr1)
sd.wait()  # Wait until the audio is finished playing