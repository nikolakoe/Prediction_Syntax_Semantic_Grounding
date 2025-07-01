import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# get EEG channel information from MEG and set montage
def create_montage(subject, meg, ch_names):
    matlab_chnames = np.load('Chan_Pos/matlab_order.npy')
    if subject > 9:
        path = 'Chan_Pos/chan_pos_' + str(subject) + '.xlsx'
    else:
        path = 'Chan_Pos/chan_pos_0' + str(subject) + '.xlsx'
    df = pd.read_excel(
        path,
        names=['x', 'y', 'z'], header=None)
    df = df.reindex(columns=['y', 'x', 'z'])
    df = np.array(df)
    ch_pos = []
    for i in raw.ch_names:
        pos = list(df[np.where(matlab_chnames == str(i))[0]][0])
        ch_pos.append(pos)
    ch_pos = np.array(ch_pos)[:64]
    pos = meg.get_positions()
    ch_pos = dict((ch_names[i], ch_pos[i]) for i in range(0, 64))
    mon = mne.channels.make_dig_montage(ch_pos=ch_pos, nasion=pos['nasion'], lpa=pos['lpa'], rpa=pos['rpa'])

    return mon

# find bad electrodes
def find_bad_channels(raw, num):
    df = raw.to_data_frame(picks="eeg", scalings=dict(eeg=1e6))  # 45000=5min until 63000=7min
    var_ch = []
    eeg_ch = raw.info.ch_names[:-2]
    for i in eeg_ch:
        var = np.var(df[i])
        var_ch.append(int(var))
    var_ch = np.array(var_ch)
    # bad electrodes are all with a variance higher than 4 times the median variance
    median = np.mean(np.sort(var_ch)[16:49]) * 4
    zer = np.argwhere(var_ch == 0)
    hig = np.argwhere(var_ch > median)
    bad_channels = []
    for i in zer:
        bad_channels.append(eeg_ch[int(i)])
    for i in hig:
        bad_channels.append(eeg_ch[int(i)])
    print("Bad channels:", bad_channels)
    return bad_channels

def find_bad_eog(raw, ica, num, thresh):
    if num == 29:   #subject 29 misses EOG channel
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name="Fp1")
    else:
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name="EOG")

    # x = np.arange(0, 30)
    # plt.figure(figsize=(15, 10))
    # plt.bar(x, np.abs(eog_scores))
    # plt.xlabel("ICA Components", fontsize=22)
    # plt.ylabel("score", fontsize=22)
    # plt.hlines(thresh, -1, 31, color="red")
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.savefig("EOG_Prob" + str(num) + ".png", dpi=150, bbox_inches="tight")
    # plt.close()
    indexes = np.arange(0, 30)
    # select indices with EOG score higher than threshold
    eog_ics = indexes[np.abs(eog_scores) > thresh]
    return eog_ics


def find_bad_ecg(raw, ica, num, thresh):
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name="EKG")
    # x = np.arange(0, 30)
    # plt.figure(figsize=(15, 10))
    # plt.bar(x, np.abs(ecg_scores))
    # plt.xlabel("ICA Components", fontsize=22)
    # plt.ylabel("score", fontsize=22)
    # plt.hlines(thresh, -1, 31, color="red")
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.savefig("ECG_Prob" + str(num) + ".png", dpi=150, bbox_inches="tight")
    # plt.close()
    indexes = np.arange(0, 30)
    # select indices with ECG score higher than threshold
    ecg_ics = indexes[np.abs(ecg_scores) > thresh]
    return ecg_ics


# resample EEG data to align with the MEG data
def resample_data(eeg, data_len, num):
    l = []
    for a in eeg.annotations:
        if a['onset'] != 0.0:
            l.append(a['onset'])
    print(l)
    start = l[2]
    stop = l[6]
    if num == 24:   #subject 24 has more trigger
        start = l[2]
        stop = l[21]

    eeg_prep = eeg.crop(tmin=start, tmax=stop)
    sample_freq = len(data_len) / (stop - start)
    eeg_prep.resample(sample_freq)

    actual = raw.copy()
    # downsample to 200Hz
    actual.resample(200)
    actual_info = actual.info

    data_eeg, time_eeg = eeg.get_data(return_times=True)
    eeg_raw = mne.io.RawArray(data_eeg, actual_info)
    return eeg_raw


if __name__ == "__main__":

    ac_time = ["0", "20231020_1157", "20231026_1437", "20231027_1423", "20231030_1417", "20231102_1422", "20231103_1506", "20231107_1414", "20231109_1305", "20231110_1458", "20231113_1442", "20231115_1440", "20231121_1422", "20231124_1409", "20231129_1438", "20231130_1513", "20231206_1405", "20231212_1423", "20231213_1419", "20231215_1413", "20231218_1428", "20231219_1427", "20231220_1443", "20240115_1420", "20240116_1406", "20240122_1407", "20240123_1411", "20240124_1350", "20240126_1011", "20240126_1425", "20240129_1410", "20240130_0943", "20240131_1103", "20240131_1505"]
    acqtime_meg = ["0", "20.10.23_1059", "26.10.23_1521", "27.10.23_1505", "30.10.23_1452", "02.11.23_1459", "03.11.23_1541", "07.11.23_1501", "09.11.23_1350", "10.11.23_1502", "13.11.23_1520", "15.11.23_1517", "21.11.23_1505", "24.11.23_1456", "29.11.23_1514", "30.11.23_1549", "06.12.23_1450", "12.12.23_1459", "13.12.23_1507", "15.12.23_1459", "18.12.23_1519", "19.12.23_1502", "20.12.23_1558", "15.01.24_1447", "16.01.24_1454", "22.01.24_1454", "23.01.24_1449", "24.01.24_1444", "26.01.24_1015", "26.01.24_1438", "29.01.24_1455", "30.01.24_1030", "31.01.24_1151", "31.01.24_1551"]

    # loop through subjects
    for subject in range(1, 34):
        # load EEG data
        if subject > 9:
            vhdr_file = '...EEG/raw/nc_lin_' + str(subject) + '_eeg_conv/' + ac_time[
                subject] + '.vhdr'
        else:
            vhdr_file = '...EEG/raw/nc_lin_0' + str(subject) + '_eeg_conv/' + ac_time[subject] + '.vhdr'
        raw = mne.io.read_raw_brainvision(vhdr_file, misc='auto', verbose=False)
        raw.load_data()
        # load MEG data
        if subject > 9:
            meg_datalen_path = "...MEG/raw/nc_lin_" + str(subject) + "/ling_10/" + \
                               acqtime_meg[subject] + "/1/c,rfhp1.0Hz"
        else:
            meg_datalen_path = "...MEG/raw/nc_lin_0" + str(subject) + "/ling_10/" + acqtime_meg[subject] + "/1/c,rfhp1.0Hz"

        meg_datalen = mne.io.read_raw_bti(meg_datalen_path, preload=True, rename_channels=False)

        meg_datalen.save("meg_dig.fif", tmax=10, overwrite=True)
        meg = mne.channels.read_dig_fif("meg_dig.fif")

        # set Electrooculogram (EOG) and Electrocardiogram (ECG) channels
        raw.set_channel_types({'EOG':'eog'})
        raw.set_channel_types({'EKG':'ecg'})
        mon = create_montage(subject, meg, np.array(raw.ch_names))
        raw.set_montage(mon)

        # find bad electrodes and interpolate the signals
        bad_channels = find_bad_channels(raw, subject)
        raw.info['bads'] = bad_channels
        raw = raw.interpolate_bads(reset_bads=True)

        # bandpass filter
        raw = raw.filter(1, 20)
        # downsample signal to 200Hz
        raw = raw.resample(200, npad="auto")

        # calculate Independent Component Analysis (ICA)
        ica = mne.preprocessing.ICA(n_components=30, random_state=97, method="fastica")
        ica.fit(raw)
        # find Independent Components (ICs) correlating with EOG and ECG channels
        iceog = find_bad_eog(raw, ica, subject, 0.1)
        icecg = find_bad_ecg(raw, ica, subject, 0.1)
        # exlcude them and the first ICs (the ones with the highest variance)
        exclude = np.concatenate(([0, 1], iceog, icecg))
        exclude = np.unique(exclude)
        print(subject, ": Exclude ICs: ", exclude)
        ica.exclude = exclude
        ica_raw = raw.copy()
        # reconstruct signal
        ica.apply(ica_raw)

        meg_datalen.pick(picks=["RESPONSE"])
        events = mne.find_events(meg_datalen, stim_channel=["RESPONSE"], output="onset")

        # crop MEG data
        start_ = events[2][0]
        stop_ = events[6][0]
        if subject == 24:
            start_ = events[2][0] #subject 24
            stop_ = events[22][0]
        df_meg = meg_datalen.to_data_frame()
        start = df_meg.loc[int(start_)]["time"]
        stop = df_meg.loc[int(stop_)]["time"]
        meg_datalen.crop(tmin=start, tmax=stop)
        meg_datalen.resample(sfreq=200)
        df_meg = meg_datalen.to_data_frame()
        # align EEG data with MEG data
        eeg_raw = resample_data(ica_raw, df_meg, subject)

        # save preprocessed EEG data
        fname = "Prob" + str(subject) + "_EEGdata_raw.fif"
        eeg_raw.save(fname, overwrite=True)
