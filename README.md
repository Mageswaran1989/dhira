# dhīra — a very learned and gentle scholar

A small practise framework created while working with Kaggle Quoa dataset on Question Similarity.

Following git repo "https://github.com/nelson-liu/paraphrase-id-tensorflow", is the base reference for my work.

Where I have made changes like using spaCy as NLP backend. 

## Why dhīra?
- Why not [Keras](https://github.com/fchollet/keras/)? Simple! Keras is a birds view from a 10000ft!!!
- Abstract common TensorFlow environment setup and concentrate more on modeling from academic papers
- Create custom Dataset classes and reuse them across models
- To write custom models from scratch and understand the model from an Enginering perspective
- Explore the possibility to merge Tensorflow and Embedded world from where I started!

### How to run?
- Check example folder

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

### [Awesome DeepLearning](https://github.com/ChristosChristofidis/awesome-deep-learning)
### [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow)
### [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)  

### How to clone this repo?
`git clone --recursive --jobs 8 https://github.com/Mageswaran1989/dhira`
`cd dhira/`
`git pull --recurse-submodules`

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