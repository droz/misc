# This is testing how well a speaker array, where the delay is controlled for each speaker,can
# emit localized sound.
import numpy as np
import soundfile as sf
import sounddevice as sd

# Speed of sound, in m/s
SPEED_OF_SOUND_M_S = 343.0

# Load audio data from file and write it as a numpy array
track1_file = "/Users/droz/Documents/GitHub/misc/audio/samples/subject1.wav"
track2_file = "/Users/droz/Documents/GitHub/misc/audio/samples/subject1.wav"

track1, sr1 = sf.read(track1_file)
track2, sr2 = sf.read(track2_file)
# Ensure both tracks have the same sample rate
if sr1 != sr2:
    raise ValueError("Sample rates of the two tracks do not match.")

# The location of the two points we are trying to project to
target1 = np.array([1, 0, 5])  # meters
target2 = np.array([-1, 0, 5])  # meters

# This is the location of each speaker in the array. They are arranged on a regular grid.
num_speakers_x = 20
num_speakers_y = 20
speaker_spacing = 0.1  # meters
speakers_x = np.linspace(-num_speakers_x / 2 * speaker_spacing, num_speakers_x / 2 * speaker_spacing, num_speakers_x)
speakers_y = np.linspace(-num_speakers_y / 2 * speaker_spacing, num_speakers_y / 2 * speaker_spacing, num_speakers_y)
speakers_x, speakers_y = np.meshgrid(speakers_x, speakers_y)
speakers_x = speakers_x.flatten()
speakers_y = speakers_y.flatten()
speakers_z = np.zeros_like(speakers_x)  # Assuming all speakers are at z=0
speakers = np.vstack((speakers_x, speakers_y, speakers_z)).T

# Calculate the delay in seconds for each speaker based on the distance to the target point
def calculate_delays(speakers, target, speed_of_sound):
    delays = np.zeros(speakers.shape[0])
    for i, speaker in enumerate(speakers):
        distance = np.linalg.norm(speaker - target)
        delays[i] = distance / speed_of_sound
    return delays

delays1 = calculate_delays(speakers, target1, SPEED_OF_SOUND_M_S)
delays2 = calculate_delays(speakers, target2, SPEED_OF_SOUND_M_S)

# Create a delayed version of the audio track for each speaker and sum them up
def apply_delays(track, delays, sample_rate):
    delayed_tracks = []
    # First we pad all the start of all the tracks
    for delay in delays:
        delay_samples = int(delay * sample_rate)
        # Create a zero-padded version of the track with the delay applied
        delayed_track = np.zeros(len(track) + delay_samples)
        delayed_track[delay_samples:] = track
        delayed_tracks.append(delayed_track)
    # Now we should pad the end of the tracks to make them all the same length
    max_length = max(len(t) for t in delayed_tracks)
    for i in range(len(delayed_tracks)):
        if len(delayed_tracks[i]) < max_length:
            delayed_tracks[i] = np.pad(delayed_tracks[i], (0, max_length - len(delayed_tracks[i])), 'constant')
    # Finally, we sum all the delayed tracks together
    return np.sum(delayed_tracks, axis=0) / len(delays)

random_delays1 = np.random.uniform(0, 0.1, size=delays1.shape)  # Randomize delays for testing

delayed_track1 = apply_delays(track1, delays1, sr1)
random_track1 = apply_delays(track1, random_delays1, sr1)




# Playback the audio tracks
sd.play(track1, samplerate=sr1)
sd.wait()  # Wait until the audio is finished playing
sd.play(delayed_track1, samplerate=sr1)
sd.wait()  # Wait until the audio is finished playing
sd.play(random_track1, samplerate=sr1)
sd.wait()  # Wait until the audio is finished playing