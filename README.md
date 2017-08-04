# dhīra — a very learned and gentle scholar

A small practise framework created while working with Kaggle Quoa dataset on Question Similarity.

Following git repo "https://github.com/nelson-liu/paraphrase-id-tensorflow", is the base reference for my work.

Where I have made changes like using spaCy as NLP backend. 

Why not [Keras](https://github.com/fchollet/keras/)?
- Simple! Keras is a birds view from a 10000ft!!!
- Abstract common TensorFlow environment setup and concentrate more on modeling from academic papers
- Create custom Dataset classes and reuse them across models
- To write custom models from scratch and understand the model from an Enginering perspective


### How to run?
- Check example folder

## Explorations

`exploration/` folder contains my reference materials and some of the best `Git repos` out there on: 
- Machine Learning
    - Udacity
        - https://github.com/udacity/machine-learning
- Deep Learning
    - Keras
        - https://github.com/leriomaggio/deep-learning-keras-tensorflow
- Maths for Engineers  

### [Awesome DeepLearning](https://github.com/ChristosChristofidis/awesome-deep-learning)
### [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow)


### How to clone this repo
`git clone --recursive --jobs 8 https://github.com/Mageswaran1989/dhira`

**Sub modules notes:**
- Add submodule and define the master branch as the one you want to track  
`git submodule add -b master [URL to Git repo]`     
`git submodule init` 

- update your submodule --remote fetches new commits in the submodules 
and updates the working tree to the commit described by the branch  
 `git submodule update --remote`