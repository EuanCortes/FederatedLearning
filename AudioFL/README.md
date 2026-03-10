# Federated learning over AudioMNIST

## Repo structure

The repo is divided into 4 main ipynb files, and a folder with source file that are used in all of them. 

1. test_base.ipynb => to test the setup on a single CNN before training bigger networks
2. FL_sound.ipynb => Basic setup for a Federated learning training over the dataset. The client dataset are close to i.i.d. and they are no personalization layer being implemented.
3. FL_sound_non_iid.ipynb => Setup changing the client data distribution to non-iid. Gives a performance comparison with the standard setup.
4. FL_sound_personnalized => Setup changing the structure of the network to have a personalization layer available. Gives a performance comparison with the two other setup.

NB: You need to run the files to have a saved model in order to successfully have the comparison available.

## Some information about the dataset
*A complete description can be found [here](https://github.com/soerenab/AudioMNIST)*.
The dataset was first used inside **AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark** ([https://www.sciencedirect.com/science/article/pii/S0016003223007536](https://www.sciencedirect.com/science/article/pii/S0016003223007536)).

Note that their article include some training and recording scripts but we mainly use a modified version of the preprocessing scripts and the dataset itself in this project.

* The dataset consists of 30000 audio samples of spoken digits (0-9) of 60 different speakers. 
* There is one directory per speaker holding the audio recordings. 
* Additionally "audioMNIST_meta.txt" provides meta information such as gender or age of each speaker.

The audio files are resampled to have a final length of 8000 data points ~2000Hz


## Prerequisite

The required pip package version can be found in requirements.txt, I suggest you use an virtual environment to better manage the package amongst your project. Be careful with the version of pytorch as the one in the requirements in using cuda 12.8. you might need an older or more recent version of pytorch. 
If you don't know what cuda is, executes this lines:
```bash
python -m venv .venv  
.venv/scripts/activate
pip install -r "AudioFL/requirements.txt"
pip uninstall torch torchvision torchaudio
pip install torch torchvision
```

After that, you need to preprocess the data from the dataset.

```bash
cd AudioFL
python preprocess.py \
    --source ../AudioMNIST/data \
    --destination ./preprocessed_data \
    --meta ../AudioMNIST/data/audioMNIST_meta.txt
```

You can change the 
And then move the content of the resulting folder to AudioFL/preprocessed_data/

You should be good to go !
