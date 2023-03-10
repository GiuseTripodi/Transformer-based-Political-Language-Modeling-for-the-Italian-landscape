# About the project

The repository contains the code developed for the master's thesis work of student Giuseppe Tripodi. The aim of the project was to assess the consistency of Italian politicians. The proposed work was created with two main objectives: 
  - the first is to assess the coherence of the content shared on social networks by Italian politicians
  - the second is to evaluate the relationship between the social content of politicians and the proposed electoral programmes for the 2022 general election


The data used correspond to tweets and transcripts of videos available on platforms such as Twitter and YouTube, respectively. On these data, different Language Models based on Transformers were trained and used with the aim of solving Text Classification and Sentence Similarity tasks. 


More information about the project itself and the results can be found on the file *Project presentation*.

## Structure

- **data**: this folder contains the dataset used in the different stages of the project


- **Develop**: this folder includes all scripts which are used during the implementation 
  - data_extraction: This folder contains all the scripts needed to extract the data used for the project.
  - data_modelling: This folder contains all the scripts and functions used for the data modelling phase. 
  - data_visualization_and_statistics: This folder contains some methods to carry out the display of results and some statistics on the data under consideration.
  - general_scripts: This folder contains some random script used during the project implementation
  - inference: This folder contains the methods needed to perform inference on the data under consideration. 
  - sentence_similarity: This folder contains the scripts needed to perform sentence similarity between sentences and also the methods used to display the results of the analysis.
  - training: This folder contains the script used to train the models.

- **notebooks:**: This folder includes all the notebooks used both to make inferences about the different data used for the project.
  - data-preprocessing: The notebook contains the preprocessing part of the project for the various types of data.
  - fine-tune-models: The notebook contains the model training part of the project. The models are trained for the different tasks.
  - inference-on-electoral-programs: The notebook contains the inference part using the model fine-tuned on speeches and tweets and used to do inference on the extracted phrases of the electoral programs.
  - inference-on-speeches-and-tweets: The notebook contains the inference part using the model fine-tuned on speeches and tweets and used to do inference on speeches and tweets.
  - sentencesimilarity: The notebook contains the sentence similarity part of the project.

- **README.md**: this file gives a brief description of the repository (e.g. folder structure, install guide, how to start it, ...)
- **requirements.txt**: this file includes all mandatory libraries (also add specific versions if necessary)

## Requirements

The project requirements are described in the **requirements.txt** file.


The main requirements are:
- Python 3.9
- Tensorflow
- [W&B](https://wandb.ai/) it is recommended to use w&b to save the trained models and to use them when needed.

### Built With

* [TensorFlow](https://www.tensorflow.org/)
* [Scikit learn](https://scikit-learn.org/stable/)
* [Keras](https://keras.io/tes)
* [W&B](https://wandb.ai/)

<!-- GETTING STARTED -->
# Getting Started

To get a local copy up and running follow these simple example steps.

## Prerequisites

Install the following libraries.

* requirements.txt
  ```sh
    pip install -r requirements.txt
  ```

## Installation


1. Clone the repo
   ```sh
   git clone [https://github.com/GiuseTripodi/Image_Text_Analysis.git](https://github.com/GiuseTripodi/Transformer-based-Political-Language-Modeling-for-the-Italian-landscape.git)


<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- CONTACT -->
## Contact

Giuseppe Tripodi - [@giuseppetripod3](https://twitter.com/giuseppetripod3) - giuseppetripodi1@outlook.it - [LinkedIn](https://www.linkedin.com/in/giuseppe-tripodi-unical/)
