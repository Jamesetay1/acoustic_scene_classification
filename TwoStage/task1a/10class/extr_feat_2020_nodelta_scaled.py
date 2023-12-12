import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool

# For 2019
# prepend_path = '/work/user_data/jtaylor/data/acoustic_scene_classification/RFR-CNN-2019/datasets/TAU-urban-acoustic-scenes-2019-development/'
# file_path = f'{prepend_path}'
# csv_file = f'{prepend_path}evaluation_setup/fold1_evaluate.csv'
# output_path = 'features_2019/logmel128_scaled'

# For 2020 Data
# file_path = '/work/user_data/jtaylor/data/acoustic_scene_classification/RFR-CNN-2019/datasets/TAU-urban-acoustic-scenes-2020-mobile-development/'
# csv_file = f'evaluation_setup_v2/fold1_evaluate.csv'
# output_path = 'features/logmel128_scaled'

# Cochlscene data
# prepend_path = '/work/user_data/jtaylor/data/acoustic_scene_classification/data/CochlScene/'
# file_path = f'{prepend_path}Val/'
# csv_file = f'{prepend_path}val_fold.tsv'
# output_path = '/work/user_data/jtaylor/data/acoustic_scene_classification/data/CochlScene/features/logmel128_scaled_v2/'

# scene data
prepend_path = '/work/user_data/jtaylor/data/acoustic_scene_classification/data/scenes/'
file_path = f'{prepend_path}Train/'
csv_file = f'{prepend_path}train_fold.tsv'
output_path = '/work/user_data/jtaylor/data/acoustic_scene_classification/data/scenes/features/logmel128_scaled_30/'


################################
######################################
################################


feature_type = 'logmel'
sr = 44100
duration = 30
num_freq_bin = 128
num_fft = 2048
hop_length = int(num_fft / 2)
num_time_bin = int(np.ceil(duration * sr / hop_length))
num_channel = 1
print(f'num time bins: {num_time_bin}')

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_df = pd.read_csv(csv_file, sep='\t', encoding='ASCII')
wavpath = data_df['filename'].tolist()


for i in range(len(wavpath)):
    print(wavpath[i])
    stereo, _fs = sound.read(file_path + wavpath[i], stop=duration*sr)
    
    print(stereo.shape)
    #Check if data is stereo, make it mono if so
    if len(stereo.shape) > 1 and stereo.shape[1] == 2:
        stereo = librosa.to_mono(stereo.T)
        
    print(stereo.shape)
    logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
    logmel_data[:,:,0]= librosa.feature.melspectrogram(stereo[:], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)

    logmel_data = np.log(logmel_data+1e-8)
    

    feat_data = logmel_data
    feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
    feature_data = {'feat_data': feat_data,}
    
    cur_file_name = output_path + os.path.basename(wavpath[i])[:-3] + feature_type
    #cur_file_name = output_path + wavpath[i][5:-3] + feature_type
    pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        

