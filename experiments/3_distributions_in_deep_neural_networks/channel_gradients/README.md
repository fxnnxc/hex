ResNet Channel wise feature and gradient norm 



```
# We squeezed the first dim

# Forward In
[torch.Size([3, 224, 224]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7])]
-----------
# Forward Out
[torch.Size([64, 112, 112]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7])]
-----------
# Backward In
# --> How much information ech channel obtains
[torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), 🔖torch.Size([3, 224, 224])]
-----------
# Backward Out
# --> How much information each channel outputs
[torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([512, 7, 7]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([256, 14, 14]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([128, 28, 28]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 56, 56]), torch.Size([64, 112, 112])]


```