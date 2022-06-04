
# SP-CSP
SP-CSP is a [MMdetection](https://github.com/open-mmlab/mmdetection) based repository, that is an anchor-free pedestrian detector. We provide pre-trained models and benchmarking of several detectors on different pedestrian detection datasets. Additionally, we provide processed annotations and scripts to process the annotation of different pedestrian detection benchmarks. 

### Installation
We refer to the installation and list of dependencies to [installation](INSTALL.md) file.
Clone this repo and follow [installation](INSTALL.md).  (Please download the pre-trained models from the table in the readme.md). 


### Datasets Preparation
* We refer to [Datasets preparation file](Datasets-PreProcessing.md) for detailed instructions


# Benchmarking 

### Benchmarking of pre-trained models on CityPersons dataset. Our results on Reasonable and Partial are both state-of-the-art.
|    Detector                | Dataset   | Backbone| Reasonable  | Heavy    | Bare | Partial |
|--------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [SP-CSP](https://drive.google.com/file/d/1hbl_0TbBabe6MCkKEYHdX2_TkUiVEwUn/view?usp=sharing) | CityPersons        | HRNet-40 | 8.7    |   46.1   |   6.0   |   7.4   |

# Getting Started

### Training

- [x] single GPU training
- [x] multiple GPU training

Train with single GPU

```shell
python tools/train.py ${CONFIG_FILE}
```

Train with multiple GPUs
```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

For instance training on CityPersons using single GPU 

```shell
python tools/train.py configs/elephant/cityperson/csp_hr40_gc.py
```

Training on CityPersons using multiple(7 in this case) GPUs 
```shell
./tools/dist_train.sh configs/elephant/cityperson/csp_hr40_gc.py 7  
```

### Testing

- [x] single GPU testing
- [x] multiple GPU testing

Test can be run using the following command.

```shell 
python ./tools/TEST_SCRIPT_TO_RUN.py PATH_TO_CONFIG_FILE ./models_pretrained/epoch_ start end\
 --out Output_filename --mean_teacher 
```

For example for CityPersons inference can be done the following way

1) Download the pretrained [CityPersons](https://drive.google.com/file/d/1hbl_0TbBabe6MCkKEYHdX2_TkUiVEwUn/view?usp=sharing) model and place it in the folder "models_pretrained/".
2) Run the following command:

```shell 
python ./tools/test_city_person.py configs/elephant/cityperson/csp_hr40_gc.py ./models_pretrained/epoch_ 81 82\
 --out result_citypersons.json --mean_teacher 
```
### References

* [Pedestron](https://openaccess.thecvf.com/content/CVPR2021/papers/Hasan_Generalizable_Pedestrian_Detection_The_Elephant_in_the_Room_CVPR_2021_paper.pdf)

