
# 🔖 Check out the result in format [paper](assets/paper.pdf)

---

# 1. Distribution Analysis - initialization test


Goal : Visualize the distribtuion of features for CNN and gradients in the convolution layers. 

Features 
1. various initialization techniques are used 
   * Kaiming Normal
   * Kaiming Uniform 
   * Xavier Normal
   * Xavier Uniform 
   * Orthogonal
   * Ones 
   * Zeros
2. Compare the initial and the final trained CNN on CIFAR10. 
    See [model.py](src/model.py) for model descript


---

We give the weight distribtuion before and after training for various initialization schemes
<img src="weight.png">


---

We also give the feature distribtuion before and after training for various initialization schemes and input types
<img src="feature.png">



### Reproduce the results 

```bash 
# step 1
cd initialization_test
bash shell/train.sh 

# step 2
bash shell/feature.sh 

# step 3 
Run the notebook
```


# 2. Channel statistics 

Gradient is the tool to analyze the behavior of a model. 

We define two types of gradient statistics for a channel 

* *Absoprtion* : How much gradient comes to the channel?
* *Emission* : How much gradient goes out from the channel?

---

We give the *Ab* and *Em* of ResNet models  pretrained on ImageNet1K

<img src="ab_em.png" >


### Reproduce the results 



```bash
cd channel_gradients 
# step 1
bash run_all.sh

# step 2
Run the notebook

```

