# LEAD
## Environment
LEAD is implemented under the following development environment:

```bash
numpy==1.22.3
dgl==1.1.1
recbole==1.1.1 
pytorch>=1.7.0 
python>=3.7.0
```
## Datasets link and brief introduction
* [Amazon](http://jmcauley.ucsd.edu/data/amazon/):
  Amazon Review Data includes reviews (ratings, text, helpfulness votes) and product metadata (descriptions, category information, price, brand, and image features), which includes a previous version in 2014 and an updated version in 2018. 
* [Yelp](https://www.yelp.com/dataset):
This dataset was collected from [Yelp](https://www.yelp.com/). The Yelp dataset is a subset of our businesses, reviews, and user data for use in personal, educational, and academic purposes. 
* [Steam](https://github.com/kang205/SASRec):
This dataset is reviews and game information from Steam, which contains 7,793,069 reviews, 2,567,538 users, and 32,135 games. 
## Usage
Our framework depends on the Recboles implementation.
In order to use RecBole, you need to convert these original datasets to the atomic file
which is a kind of data format defined by RecBole.

We provide two ways to convert these datasets into atomic files:

1. Download the raw dataset and process it with conversion tools we provide in this repository. Please refer to [conversion tools](https://github.com/RUCAIBox/RecDatasets/tree/master/conversion_tools).

2. Directly download the processed atomic files. [Baidu Wangpan](https://pan.baidu.com/s/1p51sWMgVFbAaHQmL4aD_-g) (Password: e272), [Google Drive](https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj?usp=sharing).

3. Using the program in this repositories. You can find it in `RecSysDatasets/conversion_tools`.

## Profile Generation and Semantic Representation Encoding

Firstly, we need to complete the following three steps.
- Install the openai library `pip install openai`
- Prepare your **OpenAI API Key**
- Enter your key on these files: `LEAD\generation\amazon_{item/user/emb}\.py`.

Then, here are the commands to generate the desired output with examples:

  - **User/Item Profile Generation**:

    ```python LEAD/data/{user\item}_profile_generate.py```   

  - **User/Item Auxiliary information Generation**:

    ```python AACF/generation/amazon_{usr\itm}.py```

  - **Semantic Representation**:

    ```python AACF/generation/amazon_emb.py```

Prompt file is stored in `LEAD\instruction` and can be modified according to different data sets.

## Quick Start
If you want to implement CGAN network alone:
```
cd RECGAN\cgan
python run.py
```
Running recommendation system:
```
cd Recbole\Recbole-Gnn\
```
If you want to run the backbone model:
```
python run_base.py
```
If you want to run LEAD:
```
python run_LEAD.py
```
The default model is `LightGCN` and the default data set is `Amazon_Books`.
Parameters can be modified through the command line or in a file. Supported models are: `LightGCN, LightGCL, SGL, NCL, DGCF, NGCF` Data sets are: `Amazon, Steam, Yelp`

