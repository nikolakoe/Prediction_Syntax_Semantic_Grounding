import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import correlate
import pandas as pd

def correlate_signals(data_meg, data_stimuli, xmax, samp_freq, n):
    data_sample = np.interp(np.arange(0, len(data_stimuli), (len(data_stimuli) / xmax) / samp_freq),
                            np.arange(0, len(data_stimuli)), data_stimuli)
    # correlate stimuli channel and audio signal
    corr = correlate(data_meg['X2'], data_sample, "valid")

    if n > 4:
        corr[:samp_freq*10*60] = 0
    else:
        corr[:samp_freq*4*60] = 0
    # get shift at point of strongest correlation
    shift = np.where(corr == min(corr))[0][0]
    data_shift = [None] * (shift+len(data_sample))
    data_shift[shift::] = 0.5*data_sample
    # plotting to ensure the correlation shift
    # plt.plot(data_shift, color="red")
    # plt.plot(data_meg['X2'], color="blue")
    # plt.show()
    return shift


def get_onsets(shift, stim_channel):
    shift = stim_channel['time'].iloc[shift]
    print("Time of shift point (start of audio book):", shift)

    # get the start point for each word
    onsets = []
    for i in range(len(intervals)):
        xmin = intervals[i][0] + shift
        onsets.append(xmin)

    return onsets


def get_event_id(classes):
    # create event list with labeling of word classes
    event_id = np.zeros(classes.shape[0])
    for i in range(classes.shape[0]):
        if classes[i] == "NOUN":
            event_id[i] = 1
        elif classes[i] == "PROPN":
            event_id[i] = 2
        elif classes[i] == "VERB":
            event_id[i] = 3
        elif classes[i] == "AUX":
            event_id[i] = 4
        elif classes[i] == "ADJ":
            event_id[i] = 5
        elif classes[i] == "ADV":
            event_id[i] = 6
        elif classes[i] == "PRON":
            event_id[i] = 7
        elif classes[i] == "NUM":
            event_id[i] = 8
        elif classes[i] == "CCONJ":
            event_id[i] = 9
        elif classes[i] == "SCONJ":
            event_id[i] = 10
        elif classes[i] == "PART":
            event_id[i] = 11
        elif classes[i] == "DET":
            event_id[i] = 12
        elif classes[i] == "ADP":
            event_id[i] = 13

    return event_id

# audio signals
audio_signals = ["001_Vakuum_Suedpol_0_0.npy", "002_Vakuum_Suedpol_0_1.npy", "003_Vakuum_Suedpol_0_2.npy", "006_Vakuum_Mond_0_0.npy", "007_Vakuum_Mond_0_1.npy", "004_Vakuum_Suedpol_1_0.npy", "005_Vakuum_Suedpol_1_1.npy", "008_Vakuum_Mond_1_0.npy", "009_Vakuum_Mond_1_1.npy", "015_Vakuum_Suedpol_2_0.npy", "016_Vakuum_Suedpol_2_1.npy", "017_Vakuum_Suedpol_2_2.npy", "018_Vakuum_Mond_2_0.npy", "019_Vakuum_Mond_2_1.npy", "020_Vakuum_Mond_2_2.npy", "024_Vakuum_Suedpol_3_0.npy", "025_Vakuum_Suedpol_3_1.npy", "026_Vakuum_Mond_3_0.npy", "027_Vakuum_Mond_3_1.npy", "028_Vakuum_Mond_3_2.npy"]
count = len(audio_signals)
# vakuum chapters used as stimuli for the study
chapter = ["001", "002", "003", "006", "007", "004", "005", "008", "009", "015", "016", "017", "018", "019", "020", "024", "025", "026", "027", "028"]

# loop through each subject and save epochs for EEG and MEG data
for p in range(1, 34):
    if p == 7 or p == 13 or p == 14 or p == 18: #subjects with setup problems
        continue
    epochs_eeg_total = []
    epochs_meg_total = []

    # load data:
    path_prep_eeg = "...Prob" + str(p) + "_EEGdata_raw.fif" #preprocessed EEG data
    eeg_prepr = mne.io.read_raw_fif(path_prep_eeg, preload=True).load_data()

    path_prep_meg = "...Prob" + str(p) + "_fixed_ICA_prep_raw.fif"  #preprocessed MEG data
    meg_prepr = mne.io.read_raw_fif(path_prep_meg, preload=True).load_data()

    path_stim = "...Prob" + str(p) + "_stimuli_channel_raw.fif" #stimuli channel
    proband_stim_eeg = mne.io.read_raw_fif(path_stim, preload=True).load_data()
    stim_channel = proband_stim_eeg.to_data_frame() #convert to dataframe

    amount = 0
    n = 0
    prev = 0
    for i in range(count):
        samp_freq = 200

        # load ...Vakuum_word_boundaries.npy files for each chapter
        # first value depicts length of audio file in s
        # second value shows an array with start and end time in seconds for each word
        xmax_intervals = np.load("...Excels/" + chapter[i] + "_Vakuum_word_boundaries.npy", allow_pickle=True)
        intervals = xmax_intervals[1] # word onsets

        # load audio file of each chapter
        audio_signal = np.load(audio_signals[i])[0]
        xmax = len(audio_signal) / 44100 # get the length of the signal in seconds (sampling frequency: 44100Hz)

        # get the shift data sample between the audio signal and the stimuli channel to know when the audiobook starts
        shift = correlate_signals(stim_channel, audio_signal, xmax, samp_freq, n)

        # get the onset time point for each word in the audiobook
        onsets = get_onsets(shift, stim_channel)

        # get the word class for each word in the audiobook
        classes = np.array(pd.read_excel(chapter[i] + "_wordclass.xlsx")["Spacy"])

        # create an event array encoding the wordclasses in integer values
        event_id = get_event_id(classes)

        # get eeg events
        _, time_eeg = eeg_prepr.get_data(picks="eeg", return_times=True)
        events_eeg = np.zeros((len(onsets), 3), dtype=int) # create 3 dimensional event array
        events_eeg[:, 0] = [np.argmin(np.abs(np.array(time_eeg) - i)) for i in onsets] # datasample for corresponding onset time for each word
        events_eeg[:, 2] = event_id     # event id for each word class

        # segment eeg_prepr into labeled epochs from -1s before until 2s after word onsets
        epochs_eeg = mne.Epochs(eeg_prepr, events_eeg, tmin=-1, tmax=2, picks="eeg", preload=True,
                                reject=dict(eeg=0.0001))

        # get meg events
        _, time_meg = meg_prepr.get_data(picks="meg", return_times=True)
        events_meg = np.zeros((len(onsets), 3), dtype=int)  # create 3 dimensional event array
        events_meg[:, 0] = [np.argmin(np.abs(np.array(time_meg) - i)) for i in onsets]  # datasample for corresponding onset time for each word
        events_meg[:, 2] = event_id     # event id for each word class

        # segment meg_prepr into labeled epochs from -1 until 2s after word onsets
        epochs_meg = mne.Epochs(meg_prepr, events_meg, tmin=-1, tmax=2, picks="meg", preload=True, reject=dict(mag=4*(10**(-12))))

        # append all epochs from all chapter together to get the epochs for each subject
        epochs_eeg_total.append(epochs_eeg)
        epochs_meg_total.append(epochs_meg)
        n += 1

    # concatenate eeg and meg epochs each in one file
    eeg_total = mne.concatenate_epochs(epochs_list=epochs_eeg_total, on_mismatch="ignore")
    meg_total = mne.concatenate_epochs(epochs_list=epochs_meg_total, on_mismatch="ignore")

    # set event dictionary
    event_dict = {"NOUN": 1, "PROPN": 2, "VERB": 3, "AUX": 4, "ADJ": 5, "ADV": 6, "PRON": 7, "NUM": 8, "CCONJ": 9, "SCONJ": 10, "PART": 11, "DET": 12, "ADP": 13, "X": 0}
    eeg_total.event_id = event_dict
    meg_total.event_id = event_dict

    # save epochs for each subject
    eeg_total.save("...EEG_Prob_" + str(p) + "_epo.fif",
                overwrite=True)
    meg_total.save("...MEG_Prob_" + str(p) + "_epo.fif",
                overwrite=True)
