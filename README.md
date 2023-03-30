# CycleGAN Selfie2Anime
I implemented a simple CycleGAN to solve the problem of translating selfies into anime

## Requirements
First, download Pytorch from [pytorch.org](https://pytorch.org/)
```
git clone https://github.com/dgtoan/CycleGAN_selfie2anime.git
pip install -r requirements.txt
```

## Training
<a href="https://www.kaggle.com/code/dngton/cyclegan-selfie2anime"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
### 1. Setup the dataset
You will need to download [this dataset](https://www.kaggle.com/datasets/dngton/selfie2anime).

Or you can build your own dataset by setting up the following directory structure:

    .
    ├── dataset
    |   ├── train
    |   |   ├── A
    |   |   └── B
    |   └── val
    |   |   ├── A
    |   |   └── B

### 2. Train!
```
python train.py
```
You are free to change hyperparameters and configurations in ```config.py```

You can view the training progress and live output images by running ```tensorboard --logdir output/run/tensorboard``` in another terminal and opening [http://localhost:6006/](http://localhost:6006/) in your browser.

### 3. Result
*coming soon ...*

## Testing
```
python test.py
```
As with train, you are free to change test mode and configurations in ```config.py```

Examples of the generated outputs : *coming soon ...*

<!-- ![Real horse](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/real_A.jpg) -->
<!-- ![Fake zebra](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/fake_B.png) -->
<!-- ![Real zebra](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/real_B.jpg) -->
<!-- ![Fake horse](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/fake_A.png) -->

## References
- [Arxiv](https://arxiv.org/abs/1703.10593) - Official paper on CycleGAN
- [Code](https://github.com/aitorzip/PyTorch-CycleGAN) - Ideas comes from here and some expansions on the CycleGAN paper
- [Dataset](https://www.kaggle.com/datasets/dngton/selfie2anime) - Dataset created by me
- Mentors: [QuangTran](https://github.com/pewdspie24) & [Maybe Hieu](https://github.com/maybehieu)
