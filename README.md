# Paper of the source codes released:
WenLong Zhu, YuZhong Chen, MengYu Rao.A Rumor Detection Model Based on Dynamic Heterogeneous Graph, Journal of Chinese Computer Systems 2022.

# Dependencies:
Pytorch==1.4.0

torch==1.4.0

torch-cluster==1.5.4

torch-geometric==1.6.3

torch-scatter==2.0.3

torch-sparse==0.5.1

torchvision==0.5.0


# Datasets
The main directory contains the directories of Weibo dataset and two Twitter datasets: twitter15 and twitter16. In each directory, there are:
- twitter15.train, twitter15.dev, and twitter15.test file: This files provide traing, development and test samples in a format like: 'source tweet ID \t source tweet content \t label'
  
- twitter15_graph.txt file: This file provides the source posts content of the trees in a format like: 'source tweet ID \t userID1:weight1 userID2:weight2 ...'  

These dastasets are preprocessed according to our requirement and original datasets can be available at https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0  (Twitter)  and http://alt.qcri.org/~wgao/data/rumdect.zip (Weibo).

If you want to preprocess the dataset by youself, you can use the word2vec used in our work. The pretrained word2vec can be available at https://drive.google.com/drive/folders/1IMOJCyolpYtoflEqQsj3jn5BYnaRhsiY?usp=sharing.


# Reproduce the experimental results:
1. create an empty directory: checkpoint/
2. run script run.py 
