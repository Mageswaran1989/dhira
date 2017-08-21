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

Created with [tablesgenerator!](http://www.tablesgenerator.com/markdown_tables)

| Domain  | Application         | Research Paper                                                                                                                                                 | Dataset                                                                   | dhīra Reference                                                                             | TF/Keras | References                                                                              |
|---------|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|----------|-----------------------------------------------------------------------------------------|
| NLP     |                     |                                                                                                                                                                |                                                                           |                                                                                             |          |                                                                                         |
|         | Word2Vec            | [Glove1](https://arxiv.org/abs/1408.5882)                                                                                                                      | [Moview Review](http://www.cs.cornell.edu/people/pabo/movie-review-data/) | [model](dhira/tf/models/word2vec/glove.py)  [dataset](dhira/data/dataset/movie_review.py)   | TF       | http://cs231n.github.io/convolutional-networks/                                         |
|         |                     | [Glove2](https://arxiv.org/abs/1510.03820)                                                                                                                     |                                                                           |                                                                                             |          | http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp       |
|         |                     |                                                                                                                                                                |                                                                           |                                                                                             |          | http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/ |
|         |                     |                                                                                                                                                                |                                                                           |                                                                                             |          | Git: https://github.com/yoonkim/CNN_sentence                                            |
|         | Sentence Similarity | [Siamese](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023)                                                                           | [Quora](https://www.kaggle.com/quora/question-pairs-dataset)              | [model](dhira/tf/models/siamese/siamese_bilstm.py)  [dataset](dhira/data/dataset/quora.py)  | TF       |                                                                                         |
|         |                     | [Matching Siamease](https://www.semanticscholar.org/paper/Learning-Natural-Language-Inference-using-Bidirect-Liu-Sun/f93a0a3e8a3e6001b4482430254595cf737697fa) |                                                                           | [model](dhira/tf/models/siamese/matching_bilstm.py)  [dataset](dhira/data/dataset/quora.py) |          |                                                                                         |
|         |                     |                                                                                                                                                                |                                                                           |                                                                                             |          |                                                                                         |
| CV      |                     |                                                                                                                                                                |                                                                           |                                                                                             |          |                                                                                         |
|         | Classification      | [Convolutional Imagenet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)                                | [Cifiar](https://www.cs.toronto.edu/~kriz/cifar.html)                     | [model](dhira/tf/models/conv/cifiar_convnet.py)  [dataset](dhira/data/dataset/cifiar10.py)  |          |                                                                                         |
|         |                     |                                                                                                                                                                |                                                                           |                                                                                             |          |                                                                                         |         |                     |                                                                                                                                                                |                                                                           |                                                                                                |          |                                                                                         |  

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

### [Awesome DeepLearning](https://github.com/ChristosChristofidis/awesome-deep-learning)
### [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow)
### [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)  


# All time DL References
## Losses
## Optimization
- [An overview of gradient descent optimization
algorithms∗](https://arxiv.org/pdf/1609.04747v1.pdf)



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