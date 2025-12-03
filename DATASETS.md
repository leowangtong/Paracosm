*This is an instruction on how to install the datasets used in our experiments.*

We suggest putting all datasets under the same folder (say `$data_CIR`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like:

```
$data_CIR/
|–– CIRCO/
|–– CIRR/
|–– FASHIONIQ/
|–– GeneCIS/
```

***

# How to download datasets

Datasets list:
- [CIRCO](#circo-back_to_top)
- [CIRR](#cirr-back_to_top)
- [FASHIONIQ](#fashion-iq-back_to_top)
- [GeneCIS](#genecis-back_to_top)




### CIRCO ([Back_to_top](#how-to-download-datasets))
- Create a folder named `CIRCO` under `$dataCIR`.
- Download the CIRCO dataset following the instructions in the [**official repository**](https://github.com/miccunifi/CIRCO).
- The directory structure of CIRCO should look like:

```
├── CIRCO
│   ├── annotations
|   |   ├── [val | test].json

│   ├── COCO2017_unlabeled
|   |   ├── annotations
|   |   |   ├──  image_info_unlabeled2017.json
|   |   ├── unlabeled2017
|   |   |   ├── [000000243611.jpg | 000000535009.jpg | ...]
```

### CIRR ([Back_to_top](#how-to-download-datasets))
- Create a folder named `CIRR` under `$dataCIR`.
- Download the CIRR dataset following the instructions in the [**official repository**](https://github.com/Cuberick-Orion/CIRR).
- The directory structure of CIRR should look like:

```
├── CIRR
│   ├── train
|   |   ├── [0 | 1 | 2 | ...]
|   |   |   ├── [train-10108-0-img0.png | train-10108-0-img1.png | ...]

│   ├── dev
|   |   ├── [dev-0-0-img0.png | dev-0-0-img1.png | ...]

│   ├── test1
|   |   ├── [test1-0-0-img0.png | test1-0-0-img1.png | ...]

│   ├── cirr
|   |   ├── captions
|   |   |   ├── cap.rc2.[train | val | test1].json
|   |   ├── image_splits
|   |   |   ├── split.rc2.[train | val | test1].json
```


### Fashion IQ ([Back_to_top](#how-to-download-datasets))
- Create a folder named `FASHIONIQ` under `$dataCIR`.
- Download the Fashion IQ dataset following the instructions in
the [**official repository**](https://github.com/XiaoxiaoGuo/fashion-iq). 
- The directory structure of Fashion IQ should look like:

```
├── FASHIONIQ
│   ├── captions
|   |   ├── cap.dress.[train | val | test].json
|   |   ├── cap.toptee.[train | val | test].json
|   |   ├── cap.shirt.[train | val | test].json

│   ├── image_splits
|   |   ├── split.dress.[train | val | test].json
|   |   ├── split.toptee.[train | val | test].json
|   |   ├── split.shirt.[train | val | test].json

│   ├── images
|   |   ├── [B00006M009.jpg | B00006M00B.jpg | B00006M6IH.jpg | ...]
```



#### GeneCIS ([Back_to_top](#how-to-download-datasets))
- Create a folder named `GeneCIS` under `$dataCIR`.
- Setup the GeneCIS benchmark following the instructions in the [**official repository**](https://github.com/facebookresearch/genecis). You would need to download images from the MS-COCO 2017 validation set and from the VisualGenome1.2 dataset. 
- The directory structure of CIRR should look like:

```
├── GeneCIS
│   ├── caption
|   |   ├── change_attribute.json
|   |   ├── change_object.json
|   |   ├── focus_attribute.json
|   |   ├── focus_object.json

│   ├── val2017
|   |   ├── [000000000139.jpg | 000000000285.jpg | ...]

│   ├── Visual_Genome
|   |   ├── VG_All
|   |   |   ├── [1.jpg | 2.jpg | ...]
```




## Back to [README.md](README.md)