import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from importlib.metadata import version
from lib.eval import eval_ppl, eval_zero_shot
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
import neptune

run = neptune.init_run(
    project="xuchi/LayerWiseZgf",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5Zjc5NDhlNS04MmE5LTQzOTctOWNkOC0yMGY4YWNmNGRmNjUifQ==",
)

params = {
    "sparsity_type": "1:4",
}
run["parameters"] = params


print('cuda', torch.version.cuda)
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model_name,
            cache_dir="llm_weights",
            seqlen=2048):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = seqlen
    return model


def main():
    # 一共32层，循环33次。第一次是什么都不剪，第二次是剪第一层，以此类推
    for i in range(33):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, help='LLaMA model')
        parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
        parser.add_argument('--calib_dataset', type=str, default="c4", help='Calibration dataset')
        # parser.add_argument('--calib_dataset', type=str, default="pile", help='Calibration dataset')
        parser.add_argument('--eval_dataset', type=str, default="wikitext2", help='Evaluation dataset')
        parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
        parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length')
        parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
        parser.add_argument("--sparsity_type", type=str, default="unstructured", help="Sparsity type, choose from unstructured, 4:8, 1:4, 2:4, 3:4. \
                               Please choose from the corresponding sparsity ratio")
        parser.add_argument("--prune_method", type=str,
                            choices=["svd_finetuned", "magnitude", "ri", "wanda", "svd_ri", "svd", "sparsegpt", "ria"])
        parser.add_argument("--cache_dir", default="llm_weights", type=str)
        parser.add_argument('--save', action="store_true")
        # 这里default放自己存权重的路径，可以自己设一个
        parser.add_argument('--save_model', type=str,
                            default='/home/workingplace/xc/exercise/Relative-importance-and-activation-pruning/llm_weights',
                            help='Path to save the pruned model.')
        parser.add_argument('--semi_sparse_acc', action="store_true",
                            help="using pytorch semi sparse acceleration. Only when sparsity type is 2:4")
        parser.add_argument("--eval_zero_shot", action="store_true", help="zero-shot performance")
        parser.add_argument("--a", type=float, default=0.5, help="exponenet of activation")
        parser.add_argument("--reconstruction", action="store_true",
                            help="remaining weight reconstruction based on sparsegpt")
        parser.add_argument("--reallocation", action="store_true", help="Heuristic Channel Reallocation")
        parser.add_argument("--lsa", action="store_true", help="Linear Sum Assignment")
        parser.add_argument("--importance_score", type=str, default="sum", help="assign importance score for columns")
        parser.add_argument("--gptq", action="store_true", help="use gptq or not")
        parser.add_argument("--per_outneuron", action="store_true", help="pruning per outneuron. Wanda's tactic.")
        parser.add_argument("--test_bs", type=int, default=1, help="test batch size")
        parser.add_argument("--use_cusparselt", action="store_true")
        parser.add_argument("--layer_wise", action="store_true")
        parser.add_argument("--svd_threshold", type=float, default=1e-3)
        parser.add_argument("--fast", action="store_true")
        args = parser.parse_args()
        if args.use_cusparselt:
            SparseSemiStructuredTensor._FORCE_CUTLASS = False
        # Setting seeds for reproducibility
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)

        # Handling n:m sparsity
        prune_n, prune_m = 0, 0
        args.sparsity_type=params["sparsity_type"]
        if args.sparsity_type != "unstructured":
            prune_n, prune_m = map(int, args.sparsity_type.split(":"))
        model_name = args.model.split("/")[-1]
        print(f"loading llm model {args.model}")
        # 这里写自己存放模型的位置
        args.model = "/home/workingplace/xc/exercise/Relative-importance-and-activation-pruning/llama-2-7b-hf"
        model = get_llm(args.model, args.cache_dir, args.seqlen)
        model.eval()
        if "opt" in args.model:
            tokenizer = GPT2Tokenizer.from_pretrained(args.model, use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        device = torch.device("cuda:2")
        if "30b" in args.model or "65b" in args.model:  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
            device = model.hf_device_map["lm_head"]
        print("use device ", device)
        print(model)
        if args.sparsity_ratio != 0:
            print("pruning starts")
            from lib.prune import prune_magnitude, prune_sparsegpt, prune_ria, check_sparsity
            if args.prune_method == "wanda":
                prune_ria(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "magnitude":
                prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "sparsegpt":
                prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "ria":
                prune_ria(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, skip_layer=i)
                # 下面是原函数，上面的skip_layer是我自己加的
                # prune_ria(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "ri":
                prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "svd":
                prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "svd_ri":
                prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "svd_finetuned":
                prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

            print("*" * 30)
            sparsity_ratio = check_sparsity(args, model)
            print(f"sparsity sanity check {sparsity_ratio:.4f}")
            print("*" * 30)

        ppl_test = eval_ppl(model, tokenizer, args.eval_dataset, args.test_bs, device)
        print(f"wikitext perplexity {ppl_test}")

        run["ppl_test/ppl"].append(ppl_test)

        if args.save:
            dirname = "results/{}".format("")
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            if args.layer_wise:
                filename = f"log_{args.prune_method}_layer.txt"
            else:
                filename = f"log_{args.prune_method}.txt"
            save_filepath = os.path.join(dirname, filename)
            with open(save_filepath, "a") as f:
                print("method\tactual_sparsity\tsparsity_pattern\treallocation\timportance_score\tlsa\tppl_test",
                      file=f,
                      flush=True)
                print(
                    f"{args.prune_method}\t{sparsity_ratio:.4f}\t{args.sparsity_type}\t{args.reallocation}\t{args.importance_score}\t{args.lsa}\t{ppl_test:.4f}",
                    file=f, flush=True)

        if args.save_model:
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)

        import gc

        del model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
