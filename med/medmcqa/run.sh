for model in \
    michiyasunaga/BioLinkBERT-large \
    michiyasunaga/BioLinkBERT-base \
    microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
do
    for datafolder in data
    do
        python3 train.py --model $model --dataset_folder_name $datafolder --use_context
    done
done