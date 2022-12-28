echo '👉 Running layer specific result for all samples.  The channels are randomly selected (fixed)'  

# # ~ 34
encoder='resnet34'
for target_layer in 0 1 15 25 30 # 8
do 
num_channels=50
num_flat_samples=200
python run.py \
    --encoder $encoder \
    --target-layer $target_layer \
    --num-channels $num_channels \
    --num-flat-samples $num_flat_samples
done
