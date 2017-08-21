# dhīra — a very learned and gentle scholar

A primitive practise framework created while working with Kaggle Quoa dataset on Question Similarity.

Following git repo "https://github.com/nelson-liu/paraphrase-id-tensorflow", is the base reference for my work.


## Why dhīra?
- To model or practice any TensorFlow examples/tutorials out there, with minimal work
- To explore application based use cases with TensorFlow
- Why not [Keras](https://github.com/fchollet/keras/)? Simple! Keras is a birds view from a 10000ft!!!
- Abstract common TensorFlow environment setup and concentrate more on modeling from academic papers
- Creates custom Dataset classes and reuses them across models
- To write custom models from scratch and understand the model from an Enginering perspective

## Papers covered so far...

Created with [tablesgenerator](http://www.tablesgenerator.com/markdown_tables) and validated [here!](http://dillinger.io/)

| Domain  | Application            | Research Paper                                                                                                                                                                                                                                                                                                                                                                                        | Dataset                                                                   | dhīra Reference                                                                                                                                           | TF/Keras | References                                                                                                                                                                                                                                                                            |
|---------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NLP     |                        |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                           |                                                                                                                                                           |          |                                                                                                                                                                                                                                                                                       |
|         | Word2Vec               |  1.[Glove](https://nlp.stanford.edu/pubs/glove.pdf)   2.[Noise Contrasive Estimation](https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf)  3.[Neural Probabilistic Language Models](https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf)  4.[Graphical Models for Language Modeling](https://www.cs.toronto.edu/~amnih/papers/threenew.pdf)  |                                                                           | [model](dhira/tf/models/word2vec/glove.py)  [dataset](dhira/data/dataset/movie_review.py)  [feature](dhira/data/features/glove_feature.py)                | TF       |                                                                                                                                                                                                                                                                                       |
|         | Sentence Similarity    | [Siamese LSTM](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023)                                                                                                                                                                                                                                                                                                             | [Quora](https://www.kaggle.com/quora/question-pairs-dataset)              | [model](dhira/tf/models/siamese/siamese_bilstm.py)  [dataset](dhira/data/dataset/quora.py)  [feature](dhira/data/features/quora_feature.py)               | TF       |                                                                                                                                                                                                                                                                                       |
|         |                        | [Matching Siamease LSTM](https://www.semanticscholar.org/paper/Learning-Natural-Language-Inference-using-Bidirect-Liu-Sun/f93a0a3e8a3e6001b4482430254595cf737697fa)                                                                                                                                                                                                                                   |                                                                           | [model](dhira/tf/models/siamese/matching_bilstm.py)  [dataset](dhira/data/dataset/quora.py)  [feature](dhira/data/features/quora_feature.py)              |          |                                                                                                                                                                                                                                                                                       |
|         | Sentence Clasification | [ConvNet1](https://arxiv.org/abs/1408.5882)  [ConvNet2](https://arxiv.org/abs/1510.03820)                                                                                                                                                                                                                                                                                                             | [Moview Review](http://www.cs.cornell.edu/people/pabo/movie-review-data/) | [model](dhira/tf/models/conv/sentiment_convnet.py)  [dataset](dhira/data/dataset/movie_review.py)  [feature](dhira/data/features/movie_review_feature.py) |          | 1.http://cs231n.github.io/convolutional-networks/    2.http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp   3.http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/Git:    4.https://github.com/yoonkim/CNN_sentence |
|         |                        |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                           |                                                                                                                                                           |          |                                                                                                                                                                                                                                                                                       |
| CV      |                        |                                                                                                                                                                                                                                                                                                                                                                                                       |                                                                           |                                                                                                                                                           |          |                                                                                                                                                                                                                                                                                       |
|         | Classification         | [Convolutional Imagenet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)                                                                                                                                                                                                                                                                       | [Cifiar](https://www.cs.toronto.edu/~kriz/cifar.html)                     | [model](dhira/tf/models/conv/cifiar_convnet.py)  [dataset](dhira/data/dataset/cifiar10.py)  [feature](dhira/data/features/image_feature.py)               |          |                                                                                                                                                                                                                                                                                       |
|         |                        | [Alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)                                                                                                                                                                                                                                                                                      |                                                                           | [model](dhira/keras/alexnet.py)                                                                                                                           | Keras    | https://www.analyticsvidhya.com/blog/2017/08/10-advanced-deep-learning-architectures-data-scientists/                                                                                                                                                                                 |
|         | Recognition            | [VGGnet](https://arxiv.org/abs/1409.1556)                                                                                                                                                                                                                                                                                                                                                             |                                                                           |                                                                                                                                                           | Keras    | Code: https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py                                                                                                                                                                                                       |
|         |                        | [GoogleNet](https://arxiv.org/abs/1512.00567)                                                                                                                                                                                                                                                                                                                                                         |                                                                           |                                                                                                                                                           | Keras    | Code: https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py                                                                                                                                                                                                |
|         |                        | [ResNet](https://arxiv.org/abs/1512.03385)                                                                                                                                                                                                                                                                                                                                                            |                                                                           |                                                                                                                                                           | Keras    | Code: https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py                                                                                                                                                                                                    |
|         |                        | [ResNeXt](https://arxiv.org/pdf/1611.05431.pdf)                                                                                                                                                                                                                                                                                                                                                       |                                                                           |                                                                                                                                                           | Keras    | Code: https://github.com/titu1994/Keras-ResNeXt                                                                                                                                                                                                                                       |
|         |                        | [RCNN (Region Based CNN)](https://arxiv.org/abs/1506.01497)                                                                                                                                                                                                                                                                                                                                           |                                                                           |                                                                                                                                                           | Keras    | Code: https://github.com/yhenon/keras-frcnn                                                                                                                                                                                                                                           |
|         |                        | [YOLO (You Only Look Once)](https://pjreddie.com/media/files/papers/yolo.pdf)  Notes: Real time network! TODO!!!                                                                                                                                                                                                                                                                                      |                                                                           |                                                                                                                                                           | Keras    | Code: https://github.com/allanzelener/YAD2K                                                                                                                                                                                                                                           |
|         |                        | [SqueezeNet](https://arxiv.org/abs/1602.07360)  Notes: Must explore! TODO!!!                                                                                                                                                                                                                                                                                                                          |                                                                           |                                                                                                                                                           | Keras    | Code: https://github.com/rcmalli/keras-squeezenet                                                                                                                                                                                                                                     |
|         | Segmentation           | [SegNet](https://arxiv.org/abs/1511.00561)                                                                                                                                                                                                                                                                                                                                                            |                                                                           |                                                                                                                                                           | Keras    | Code: https://github.com/imlab-uiip/keras-segnet                                                                                                                                                                                                                                      |
|         | Others                 | [GAN](https://arxiv.org/abs/1406.2661)                                                                                                                                                                                                                                                                                                                                                                |                                                                           |                                                                                                                                                           | Keras    | 1. Code: https://github.com/bstriner/keras-adversarial     2. https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/                                                                                                                         |
              
### How to run?
- Check example folder


### How to clone this repo?
`git clone --recursive --jobs 8 https://github.com/Mageswaran1989/dhira`  
`cd dhira/`  
`git pull --recurse-submodules`  

## Explorations

`exploration/` folder contains my reference materials and some of the best `Git repos` out there on:
- Cookbooks
    - Python Pandas
        - https://github.com/jvns/pandas-cookbook
    - TensorFlow
        - https://github.com/nfmcclure/tensorflow_cookbook
- Machine Learning
    - Udacity
        - https://github.com/udacity/machine-learning
- Deep Learning
    - Keras
        - https://github.com/leriomaggio/deep-learning-keras-tensorflow
- Maths for Engineers 
    - Linear Algebra
        - https://github.com/juanklopper/MIT_OCW_Linear_Algebra_18_06
    - Statistics
        - https://github.com/AllenDowney/ThinkStats2
    - Diffrential Equations
        - https://github.com/juanklopper/Differential-Equations
-Jupyter Notebooks
    - Data Science 
        - https://github.com/donnemartin/data-science-ipython-notebooks

# All time DL References
## Optimization
- [An overview of gradient descent optimization
algorithms∗](https://arxiv.org/pdf/1609.04747v1.pdf)
## Convolutional Network
- http://cs231n.github.io/convolutional-networks/ 


### [Awesome DeepLearning](https://github.com/ChristosChristofidis/awesome-deep-learning)
### [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow)
### [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)  


-------------------------------------------------------------------------------------------------------
**Sub modules notes:**

```commandline
#add submodule and define the master branch as the one you want to track  
git submodule add -b master [URL to Git repo]     
git submodule init

#update your submodule --remote fetches new commits in the submodules 
# and updates the working tree to the commit described by the branch  
# pull all changes for the submodules
git submodule update --remote
 ---or---
# pull all changes in the repo including changes in the submodules
git pull --recurse-submodules


# update submodule in the master branch
# skip this if you use --recurse-submodules
# and have the master branch checked out
cd [submodule directory]
git checkout master
git pull

# commit the change in main repo
# to use the latest commit in master of the submodule
cd ..
git add [submodule directory]
git commit -m "move submodule to latest commit in master"

# share your changes
git push
``` 