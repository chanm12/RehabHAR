#!/bin/bash

# Define the folder path
folder_path="/coc/pcba1/mthukral3/gt/smarthome_discretization/lexicon-of-human-movements-smarthome_aruba_512emb/code/vq_cpc/models/Nov-21-2023-pretrained"

# Iterate over each file in the folder
for filename in "$folder_path"/*
do
    # Assuming you want to call a Python script and pass the filename as an argument
    python main.py --saved_model "$filename"

    # If you just want to set a shell variable, you can do it like this
    # saved_model="$filename"
    # And then use $saved_model as needed
done
