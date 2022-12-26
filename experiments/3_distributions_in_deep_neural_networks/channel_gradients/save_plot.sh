
for model in  18 34 50 101 152
do 
    for direction in 'in' 'out'
    do 
    python save_plot.py \
        --model '18' \
        --direction $direction

    done 
done 