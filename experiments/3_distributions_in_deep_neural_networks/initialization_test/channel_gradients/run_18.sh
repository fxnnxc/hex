echo '👉 Running layer specific result for all samples.  The channels are randomly selected (fixed)'  
# 

# ~ 17
encoder='resnet18' 
for target_layer in 16 # 0 1 6 9 16   #  8
do 
num_channels=50
python run.py \
    --encoder $encoder \
    --target-layer $target_layer \
    --num-channels $num_channels
done

