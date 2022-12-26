
🔖 Check out the result in [paper](assets/paper.pdf)


# Distribution Analysis - initialization test


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


# Reproduce the results 

```bash 
# step 1
cd initialization_test
bash shell/train.sh 

# step 2
bash shell/feature.sh 

# step 3 
Run the notebook
```


# Channel statistics 

```bash
cd channel_gradients 
# step 1
bash run_all.sh

# step 2
Run the notebook

```

