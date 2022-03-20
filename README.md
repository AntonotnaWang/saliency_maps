# Saliency Maps
PyTorch implementation of some saliency map methods for XAI.

**It can be used for any PyTorch pretrained models and show the saliency map of any layer.**

Please follow the tutorial in the notebook ```show_saliency_maps.ipynb```.

![saliency_maps](saliency_maps.png)

![saliency_maps_II](saliency_maps_II.png)

## update
```v0``` is the old version. The new version can be used for any PyTorch pretrained models and show the saliency map of any layer (e.g., for ```VGG19```, you set the hook layer(s) ```[features.20]``` or ```[features.10, features.20, features.30]```).