import boto3
import os
import re
import argparse
import sys
import time
from botocore.exceptions import ClientError

from transformers import LlavaOnevisionForConditionalGeneration
import torch.distributed.checkpoint as dcp

from torchtitan.logging import logger

def strip_s3_protocol(s3_path):
    """
    Removes the s3:// protocol prefix from an S3 path and returns bucket and key parts.
    
    Args:
        s3_path (str): An S3 path which may include the s3:// protocol
        
    Returns:
        tuple: (bucket_name, key_prefix)
    """
    # Remove s3:// if present
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    
    # Split into bucket and key
    parts = s3_path.split('/', 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''
    
    return bucket, prefix

def download_directory(s3, bucket, prefix, local_dir):
    """
    Download all files from an S3 directory to a local directory.
    
    Args:
        s3: boto3 S3 client
        bucket (str): S3 bucket name
        prefix (str): S3 key prefix (directory)
        local_dir (str): Local directory to download files to
    """
    # Create the local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # List all objects in the S3 directory
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    if 'Contents' not in response:
        print(f"No files found in s3://{bucket}/{prefix}")
        return
    
    total_files = len(response['Contents'])
    print(f"Downloading {total_files} files from s3://{bucket}/{prefix} to {local_dir}")
    
    # Download each file
    for i, obj in enumerate(response['Contents']):
        # Get the relative path of the file within the directory
        key = obj['Key']
        if key == prefix:  # Skip the directory itself
            continue
            
        # Create local file path
        rel_path = key[len(prefix):].lstrip('/')
        local_file_path = os.path.join(local_dir, rel_path)
        
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Download file with progress indication
        print(f"Downloading ({i+1}/{total_files}): {rel_path}")
        s3.download_file(bucket, key, local_file_path)

def get_latest_checkpoint(s3_path, local_base_dir, default_dir):
    """
    Check AWS S3 for step-based checkpoints, download the latest one to local directory,
    and return the local path.
    
    Args:
        s3_path (str): The S3 path to check for checkpoints (can include s3:// protocol)
        local_base_dir (str): Local base directory to download checkpoints to
        default_dir (str): Default directory to return if no checkpoints found
    
    Returns:
        str: Path to the local directory containing the latest checkpoint
    """
    try:
        # Parse bucket and prefix from the s3_path
        bucket_name, prefix = strip_s3_protocol(s3_path)
        
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        # Ensure prefix ends with a slash if it's not empty
        if prefix and not prefix.endswith('/'):
            prefix += '/'
            
        # List objects in the bucket with the given prefix
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            Delimiter='/'
        )
        
        # Check if there are any common prefixes (directories)
        if 'CommonPrefixes' not in response or not response['CommonPrefixes']:
            print(f"No checkpoint directories found in s3://{bucket_name}/{prefix}")
            return default_dir
        
        # Pattern to match checkpoint directories with step numbers (e.g., step-10000/)
        step_pattern = re.compile(r'.*step(\d+)/?$')
        
        # Filter directories that match the checkpoint pattern and extract step numbers
        checkpoint_dirs = []
        for common_prefix in response['CommonPrefixes']:
            dir_path = common_prefix['Prefix']
            match = step_pattern.match(dir_path)
            
            if match:
                step_number = int(match.group(1))
                checkpoint_dirs.append((dir_path, step_number))
        
        if not checkpoint_dirs:
            print(f"No checkpoint directories with step format found in s3://{bucket_name}/{prefix}")
            return default_dir
        
        # Sort by step number (highest first = latest checkpoint)
        checkpoint_dirs.sort(key=lambda x: x[1], reverse=True)
        
        # Get the latest checkpoint directory
        latest_s3_path = checkpoint_dirs[0][0]
        latest_step = checkpoint_dirs[0][1]
        print(f"Latest checkpoint found: s3://{bucket_name}/{latest_s3_path} (step {latest_step})")
        
        # Create local directory for the checkpoint
        ckpt_dirname = os.path.basename(latest_s3_path.rstrip('/'))
        local_ckpt_dir = os.path.join(local_base_dir, ckpt_dirname)
        
        # Download the checkpoint files
        print(f"Downloading checkpoint to {local_ckpt_dir}...")
        download_directory(s3, bucket_name, latest_s3_path, local_ckpt_dir)
        
        print(f"Checkpoint downloaded successfully to {local_ckpt_dir}")
        return local_ckpt_dir
    
    except Exception as e:
        print(f"Error processing checkpoints: {str(e)}")
        return default_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download latest step-based checkpoint from S3 to local directory.')
    parser.add_argument('--path', required=True, help='S3 path to check (can include s3:// prefix)')
    parser.add_argument('--local-dir', required=True, help='Local directory to download checkpoints to')
    parser.add_argument('--default-dir', default="distributed_checkpoint/", help='Default directory to use if no checkpoints found')
    
    args = parser.parse_args()
    
    # Make sure the local base directory exists
    os.makedirs(args.local_dir, exist_ok=True)
    output_dir = get_latest_checkpoint(args.path, args.local_dir, args.default_dir)
    os.environ['CHECKPOINT_DIR'] = output_dir
    print("chekcpoints: ", os.environ['CHECKPOINT_DIR'])

    if output_dir == args.default_dir:
        model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_name)

        # TODO: Convert and save the distributed checkpoint
        # Save the distributed checkpoint
        dcp.save({"model": model.state_dict()}, checkpoint_id=output_dir)
        print(f"Distributed checkpoint saved at {output_dir}")

        try:
            CONFIG_FILE = os.environ['CONFIG_FILE']
            command = f"torchrun --nproc_per_node 1 train_llava.py --job.config_file {CONFIG_FILE} --checkpoint.create_seed_checkpoint --experimental.context_parallel_degree 1"
            logger.info("Making a seed checkpoint ...")
            subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            logger.info(f"Done with creating seed checkpoint !")
        except Exception as e:
            logger.error(f"Error: {e}")

