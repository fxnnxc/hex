echo '👉 Running layer specific result for all samples.  The channels are randomly selected (fixed)'  
# 



# ~ 49
encoder='resnet50'
for target_layer in 30 39 45  # 0 5 10 15 21 30 39 45 # 8
do 
num_channels=50
num_flat_samples=200
python run.py \
    --encoder $encoder \
    --target-layer $target_layer \
    --num-channels $num_channels \
    --num-flat-samples $num_flat_samples
done

