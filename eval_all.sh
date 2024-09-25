for model in llava vipllama gemini_vision;
do
    for informed in False True;
    do
        for direct in False True;
        do
            python startotxt_infer_with_cache.py --model $model --informed $informed --direct $direct
        done
    done
done