echo "Predicting Prefill Phase..."
for batch_size in 1 2 4 8 16 32 64;
# for batch_size in $(seq 1 64);
do
    echo "Batch Size: ${batch_size}"
    for seq_len in 32 64 128 256 512 1024;
    # for seq_len in $(seq 1 1024);
    do
        echo "\tSeq Length: ${seq_len}"
        printf "\t\t"
        uv run model_analyzer.py --analyze --model_json models/transformers.models.opt.modeling_opt.OPTForCausalLM_populated.json --hardware nvidia_A800_80G_PCIe --batch_size ${batch_size} --seq_len ${seq_len} | tail -n 1
    done
done

echo "Predicting Decode Phase..."
for batch_size in 1 2 4 8 16 32 64;
do
    echo "Batch Size: ${batch_size}"
    for cache_len in 32 64 128 256 512 1024;
    do
        echo "\tCache Length: ${cache_len}"
        printf "\t\t"
        uv run model_analyzer.py --analyze --model_json models/transformers.models.opt.modeling_opt.OPTForCausalLM_populated.json --hardware nvidia_A800_80G_PCIe --batch_size ${batch_size} --seq_len 1 --cache_len ${cache_len} | tail -n 1
    done
done
