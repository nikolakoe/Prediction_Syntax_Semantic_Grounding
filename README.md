# EEG/MEG Study
## Prediction, Syntax and Semantic Grounding in the Brain and Large Language Models

This repository explains the preprocessing and methods used to analyze the EEG and MEG data. The data was recorded using combined EEG and MEG measurements while the participants listened to an hour of the science fiction audio book "Vakuum" by Phillip P. Peterson.
The word classes and onsets for each word in the audio book chapters are saved in Excels/...wordclasses.xlsx and Excel/...Vakuum_word_boundaries.npy.
The preprocessed EEG and MEG data with the according stimuli channels can be downloaded from Zenodo (EEG/MEG recordings using german audio book).
For audio signals and transcript of the german audio book contact the authors.

1. EEG data was preprocessed using *EEG_preprocess_OpenAccess.py* 
2. MEG data was preprocessed using *MEG_preprocess_OpenAccess.py*
3. For aligning the audio book with stimuli channels we used Forced Alignment *https://clarin.phonetik.uni-muenchen.de/BASWebServices/interface/WebMAUSBasic* with the audio signals and transcript to get the word onsets
4. Each word in the transcript was classified into word classes using spaCy *https://spacy.io/*
5. For segmenting the continuous data into epochs for each word and labeling each epoch with according word class use *get_word_classes_epochs_OpenAccess.py*
6. Statistical tests where done using brainstorm software: *https://neuroimage.usc.edu/brainstorm/Installation*
7. *semantic_predictability_OpenAccess.py*: calculating the semantic predictability scores for each word class (NOUN, VERB, ADJ, PROPN) with Llama 3.2 model
8. *syntactic_predictability_OpenAccess.py*: generating syntactic predictability scores using hidden representations of Llama 3.2 model and class labels of following word
