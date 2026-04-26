import pandas as pd
import json

df = pd.read_csv('synthetic_vibration_data.csv')
features = ['machine_id_encoded', 'operating_speed_rpm', 'load_percentage', 'temperature_celsius', 'rms', 'peak_to_peak', 'kurtosis', 'skewness', 'crest_factor', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'dominant_frequency', 'frequency_rms', 'entropy', 'impulse_factor', 'clearance_factor', 'band_energy_1_5kHz', 'snr_estimated']

ranges = {
    'normal': {f: [float(df[df['fault_present']==0][f].min()), float(df[df['fault_present']==0][f].max())] for f in features},
    'faulty': {f: [float(df[df['fault_present']==1][f].min()), float(df[df['fault_present']==1][f].max())] for f in features}
}
with open('stats.json', 'w') as f:
    json.dump(ranges, f)
ftsm mkus dkva ksad
