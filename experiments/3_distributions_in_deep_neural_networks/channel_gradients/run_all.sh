echo '👉 Running layer specific result for all samples.  The channels are randomly selected (fixed)'  
# 

# ~ 17
encoder='resnet18' 
for target_layer in 1 3 6 8 9 12 14 16   #  8
do 
num_channels=50
python run.py \
    --encoder $encoder \
    --target-layer $target_layer \
    --num-channels $num_channels
done

# # ~ 34
encoder='resnet34'
for target_layer in 0 3 6 10 15 20 25 30 # 8
do 
num_channels=50
num_flat_samples=200
python run.py \
    --encoder $encoder \
    --target-layer $target_layer \
    --num-channels $num_channels \
    --num-flat-samples $num_flat_samples
done

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


# ~ 100
encoder='resnet101'
for target_layer in 0 10 30 45 51 60 79 95 # 8
do 
num_channels=50
num_flat_samples=200
python run.py \
    --encoder $encoder \
    --target-layer $target_layer \
    --num-channels $num_channels \
    --num-flat-samples $num_flat_samples
done

# ~151
encoder='resnet152'
for target_layer in 0 20 50 75 98 100 125 150 # 8
do 
num_channels=50
num_flat_samples=200
python run.py \
    --encoder $encoder \
    --target-layer $target_layer \
    --num-channels $num_channels \
    --num-flat-samples $num_flat_samples
done
