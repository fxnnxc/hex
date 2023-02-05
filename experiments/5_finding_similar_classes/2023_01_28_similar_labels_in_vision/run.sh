encoder='resnet18'
for k in 9 8 7 6 5 4 3 2 
do 
    python eval_sim.py --encoder $encoder --top-k $k
done 