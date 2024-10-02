import argparse
import sys
from pathlib import Path
from lerobot.common.policies.act.modeling_act import ACTPolicy
from huggingface_hub import upload_file

def main(repo_id, checkpoint_dir):
    # Download the diffusion policy for pusht environment
    pretrained_policy_path = Path(checkpoint_dir)

    # Load the policy from the pretrained checkpoint
    policy = ACTPolicy.from_pretrained(pretrained_policy_path)

    # Push policy to the specified Hugging Face Hub repository
    policy.push_to_hub(repo_id)

    # Upload the YAML configuration file separately
    yaml_config_path = pretrained_policy_path / "config.yaml"
    upload_file(
        path_or_fileobj=yaml_config_path,
        path_in_repo="config.yaml",
        repo_id=repo_id,
        commit_message="Upload YAML configuration"
    )

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Push a pretrained ACT policy and YAML config to the Hugging Face Hub.",
        epilog="Example usage:\n"
               "  python save_policy.py --repo_id iantc104/sim_slot_insertion_3arms_zed_wrist_act "
               "--checkpoint_dir outputs/train/sim_slot_insertion_3arms_zed_wrist_act/checkpoints/014000/pretrained_model",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Add arguments
    parser.add_argument('--repo_id', type=str, required=True, help="Repository ID on Hugging Face Hub.")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Directory containing the pretrained model checkpoints.")

    # If no arguments are provided, show example command
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nExample usage:")
        print("  python save_policy.py --repo_id iantc104/sim_slot_insertion_3arms_zed_wrist_act "
              "--checkpoint_dir outputs/train/sim_slot_insertion_3arms_zed_wrist_act/checkpoints/014000/pretrained_model")
        sys.exit(1)

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.repo_id, args.checkpoint_dir)
