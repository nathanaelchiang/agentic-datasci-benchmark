import os
import argparse
from utils.json_operator import *
import shutil


def parse_arguments():
    """Parse command-line arguments for evaluation result display."""
    parser = argparse.ArgumentParser(description="For Evaluation Result Display")
    parser.add_argument("--task_id", type=str, default="bcb", choices=["bcb"],
                        help="Specify the task id, all bcb tasks by default")
    parser.add_argument("--model_id", type=str, default="gpt-4-turbo", help="Specify the model id")
    parser.add_argument("--tasks", nargs="+", default=None,
                    help="Specify exact task folders (e.g., bcb9 bcb20 ...)")

    args = parser.parse_args()
    return args


def main(all_args):
    """Load TMC results for all matching task folders and print pass@1 and average CR."""
    output_dir = 'data'
    model_id = all_args.model_id
    task_id = all_args.task_id
    all_crs = []
    all_success_rates = []

    import re

    for entry in os.scandir(output_dir):
        if not entry.is_dir():
            continue

        dir_ = entry.name

        # ONLY match bcb1, bcb2, ..., bcb123
        if not re.fullmatch(rf"{task_id}\d+", dir_):
            continue

        # now proceed exactly as before
        output_file_path = os.path.join(output_dir, dir_, f"{model_id}_tmc_results.jsonl")

        if not os.path.exists(output_file_path):
            continue

        output_datas = read_json(output_file_path)
        completed_n = len(output_datas)

        # crs = []
        # for i in range(10):
        #     if i >= completed_n:
        #         crs.append(0)
        #     else:
        #         crs.append(output_datas[i].get("cr", 0))

        crs = [entry.get("cr", 0) for entry in output_datas]

        # correct pass@1
        pass_1 = any(cr >= 1.0 for cr in crs)

        avg_cr = sum(crs) / len(crs)

        all_crs.append(avg_cr)
        all_success_rates.append(pass_1)

    # calculate pass@1 results
    pass_1_rate = sum(all_success_rates) / len(all_success_rates)
    total_avg_cr = sum(all_crs) / len(all_crs)
    print(f"--Results for {model_id} on {task_id}--\nPass@1 Rate: {pass_1_rate}\nAverage CR: {total_avg_cr}\n")
    output_file_name = f"results/{task_id}/{model_id}_results.json"
    output_js = [{"pass_1_rate": pass_1_rate, "avg_cr": total_avg_cr}]
    dump_json(output_file_name, output_js)

    #new
    from fractions import Fraction

    # Safety check
    if len(all_success_rates) == 0:
        print("No valid task folders found.")
        return

    # Pass@1
    pass_num = sum(all_success_rates)
    pass_den = len(all_success_rates)
    pass_frac = Fraction(pass_num, pass_den)
    pass_float = pass_num / pass_den

    # Average CR
    avg_cr_float = sum(all_crs) / len(all_crs)
    avg_cr_frac = Fraction(avg_cr_float).limit_denominator(1000)

    # Print
    print(f"--Results for {model_id} on {task_id}--\n")
    print(f"Pass@1 Rate: {pass_frac} ({pass_float:.4f})")
    print(f"Average CR: {avg_cr_frac} ({avg_cr_float:.4f})\n")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
