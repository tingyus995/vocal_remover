# Deep-Learning Vocal Remover
## Introduction
My own easy-to-understand implementation of the paper [Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks"](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf) using ``PyTorch`` and ``librosa``.

## Usage
### Training
* Put audio files with instrument-only track on the left channel and mixed (with vocal) track on the right channel into the ``data`` directory.

* Run train.py

### Inference
* Specify input media in ``inference.py``.
* Run ``inference.py``
* The result will be saved as ``result.wav``.