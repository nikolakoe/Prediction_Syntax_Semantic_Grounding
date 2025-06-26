import mne
from matplotlib import pyplot as plt
from scipy.signal import correlate
import numpy as np
from mne.preprocessing import ICA


def load_meg_data(path, i):
    # load eeg data with stimuli channel
    proband = mne.io.read_raw_bti(path, rename_channels=False, preload=True)

    # find trigger in "RESPONSE"-channel, to align EEG and MEG data
    events = mne.find_events(proband, stim_channel=["RESPONSE"], output="onset")
    start_ = events[2][0]
    stop_ = events[6][0]
    if i == 24: # for subject 24 there were some setup problems, which is why we have other triggers here
        start_ = events[2][0]
        stop_ = events[22][0]

    trigger_channel = proband.copy().pick_channels(['RESPONSE']).load_data()
    meg_trigger = trigger_channel.to_data_frame()
    start = meg_trigger.loc[int(start_)]["time"]
    stop = meg_trigger.loc[int(stop_)]["time"]

    proband.crop(tmin=start, tmax=stop)

    data_meg, _ = proband.get_data(return_times=True)
    # reset the start-time and end-time to the trigger before and at the end of the measurement
    proband = mne.io.RawArray(data_meg, proband.info, first_samp=0)

    del data_meg

    # stimuli channel with the audiobook as input in "X2"
    stimuli_channel = proband.copy().pick_channels(['X2']).load_data()

    # select MEG sensors
    channel_names = ["A" + str(i) for i in range(1, 249)]
    proband = proband.pick(picks=channel_names)
    # interpolate bad sensors
    proband.info["bads"] = ["A32", "A60", "A242", "A30", "A33"]
    proband = proband.interpolate_bads(reset_bads=True)

    return proband, stimuli_channel, trigger_channel

def correlateIC_ekg(sources, num, thresh):
    # read EEG file with EOG and ECG channels
    eog_ekg = mne.io.read_raw_fif("EEG_Preprocessed/Prob" + str(num) + "_EEGdata_raw.fif")
    eog_ekg = eog_ekg.pick(picks=["EOG", "EKG"])
    # resample to the same sampling frequency as the MEG data
    eog_ekg.resample(200)

    eog_ekg_df = eog_ekg.to_data_frame()

    channels = sources.info.ch_names
    sources = sources.to_data_frame()
    # correlate each independent component with the ECG signal
    corr = []
    for j in channels:
        corr_ = correlate(eog_ekg_df["EKG"], sources[j], "valid")
        corr.append(corr_)
    corr = [np.abs(c[0]) for c in corr]
    # find the max correlation value
    corrmax = np.max(corr)
    corr = [c/corrmax for c in corr]
    # append the index of all the channels with a higher correlation value than thresh
    idx = []
    for i in range(len(corr)):
        if corr[i] > thresh:
            idx.append(i)
    return idx


def correlateIC_eog(sources, num, thresh):
    # read EEG file with EOG and ECG channels
    eog_ekg = mne.io.read_raw_fif("W:/hno/science/koelblna/Vakuum_MEEG/EEG_Preprocessed/Prob" + str(num) + "_EEGdata_raw.fif")
    # the EOG channel of subject 29 had a misfunction --> use of frontal channel "Fp1" instead
    if num == 29:
        eog_ekg = eog_ekg.pick(picks=["Fp1"])
    else:
        eog_ekg = eog_ekg.pick(picks=["EOG", "EKG"])
    # resample to the same sampling frequency as the MEG data
    eog_ekg.resample(200)

    eog_ekg_df = eog_ekg.to_data_frame()

    channels = sources.info.ch_names
    sources = sources.to_data_frame()
    # correlate each independent component with the EOG signal
    corr = []
    for j in channels:
        if num == 29:
            corr_ = correlate(eog_ekg_df["Fp1"], sources[j], "valid")
        else:
            corr_ = correlate(eog_ekg_df["EOG"], sources[j], "valid")
        corr.append(corr_)
    corr = [np.abs(c[0]) for c in corr]
    # find the max correlation value
    corrmax = np.max(corr)
    corr = [c/corrmax for c in corr]
    # append the index of all the channels with a higher correlation value than thresh
    idx = []
    for i in range(len(corr)):
        if corr[i] > thresh:
            idx.append(i)
    return idx



def compute_ica_fixed(raw, num):
    # downsample data for computational efficiency
    raw.resample(200)
    # calculate ICA
    ica = ICA(n_components=50, max_iter='auto', random_state=97, method="fastica")
    ica.fit(raw)
    # get sources (Independent Components) from data
    sources = ica.get_sources(raw)

    # get all independents components that have eye or heartbeat signal in them
    idx_ekg = correlateIC_ekg(sources, num, 0.8)
    idx_eog = correlateIC_eog(sources, num, 0.8)
    sources = sources.to_data_frame()
    # concatenate the first two components (high variance) and the ECG and EOG correlated components
    exc = np.concatenate(([0, 1], np.array(idx_ekg), np.array(idx_eog)))
    exc = list(exc)
    exc = list(dict.fromkeys(exc))
    # exclude the components and reconstruct the data
    ica.exclude = exc
    ica.apply(raw)
    return ica, raw


def preprocessing(raw, num):
    # find bad meg sensors
    auto_noisy_chs, auto_flat_chs = mne.preprocessing.find_bad_channels_maxwell(
        raw, duration=120,
        verbose=True,
    )
    # concatenate sensors with flat signal or highly noisy signal
    exc = np.concatenate((np.array(auto_noisy_chs), np.array(auto_flat_chs)))
    exc = list(exc)
    exc = list(dict.fromkeys(exc))
    meg_ch = raw.pick(picks="meg")
    meg_ch = meg_ch.info.ch_names
    channel_exc = []
    for i in exc:
        if i in meg_ch:
            channel_exc.append(i)
    # mark them as bad
    raw.info["bads"] = channel_exc
    # interpolate them
    raw = raw.interpolate_bads(reset_bads=True)

    # filter data
    raw_lh = raw.copy().filter(l_freq=1, h_freq=20)
    # use ICA to extract eye and heartbeat artifacts
    ica, raw_ica = compute_ica_fixed(raw_lh, num)
    return raw_ica


acqtime = ["0", "20.10.23_1059", "26.10.23_1521", "27.10.23_1505", "30.10.23_1452", "02.11.23_1459",
               "03.11.23_1541", "07.11.23_1501", "09.11.23_1350", "10.11.23_1502", "13.11.23_1520", "15.11.23_1517",
               "21.11.23_1505", "24.11.23_1456", "29.11.23_1514", "30.11.23_1549", "06.12.23_1450", "12.12.23_1459",
               "13.12.23_1507", "15.12.23_1459", "18.12.23_1519", "19.12.23_1502", "20.12.23_1558", "15.01.24_1447",
               "16.01.24_1454", "22.01.24_1454", "23.01.24_1449", "24.01.24_1444", "26.01.24_1015", "26.01.24_1438",
               "29.01.24_1455", "30.01.24_1030", "31.01.24_1151", "31.01.24_1551"]


for i in range(1, 33):
    if i == 7 or i == 13 or i == 14 or i == 18: # subjects with setup problems (e.g. earphone fell out)
        continue

    # load MEG file
    if i > 9:
        path = "nc_lin_" + str(i) + "/ling_10/" + acqtime[i] + "/1/c,rfhp1.0Hz"
    else:
        path = "nc_lin_0" + str(i) + "/ling_10/" + acqtime[i] + "/1/c,rfhp1.0Hz"

    # load meg channels, stimuli channel and trigger channel
    data, stimuli_channel, trigger_channel = load_meg_data(path, i)

    # preprocess MEG data and save it
    prep_data = preprocessing(data, i)
    fname = "Bad_Channels/Prob" + str(i) + "_fixed_ICA_prep_raw.fif"
    prep_data.save(fname, overwrite=True)

    # resample and save stimuli channel
    stimuli_channel = stimuli_channel.resample(sfreq=200)
    st_name = "Bad_Channels/Prob" + str(i) + "_stimuli_channel_raw.fif"
    stimuli_channel.save(st_name, overwrite=True)

    # resample and save trigger channel
    trigger_channel = trigger_channel.resample(sfreq=200)
    tr_name = "Bad_Channels/Prob" + str(i) + "_trigger_channel_raw.fif"
    trigger_channel.save(tr_name, overwrite=True)
