"""Command-line interface for corpus2nouns."""

import argparse
import sys
from corpus2nouns.pipeline import CorpusNounPipeline
from corpus2nouns.publisher import publish_dataset

def main():
    parser = argparse.ArgumentParser(
        prog="corpus2nouns",
        description="High-throughput engine for extracting and reducing noun phrases from massive corpora."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # 1. extract
    extract_parser = subparsers.add_parser("extract", help="Extract noun phrases across all subsets in a directory.")
    extract_parser.add_argument("input_dir", help="Path to input directory containing subset parquet directories.")
    extract_parser.add_argument("-o", "--output-dir", required=True, help="Directory to save extracted noun parquets.")
    extract_parser.add_argument("-c", "--cache-dir", default=None, help="Directory for checkpoint cache.")
    extract_parser.add_argument("-w", "--workers", type=int, default=16, help="Number of worker processes.")
    extract_parser.add_argument("--reduce", action="store_true", help="Drop adjectives/determiners and deduplicate to pure nouns.")
    
    # 2. publish
    publish_parser = subparsers.add_parser("publish", help="Publish extracted noun dataset to Hugging Face Hub.")
    publish_parser.add_argument("output_dir", help="Path to output directory containing data/ and subsets.")
    publish_parser.add_argument("--repo-id", required=True, help="Target Hugging Face repo ID (e.g. AfriSpeech/SFT-base-en-nouns).")
    publish_parser.add_argument("--token", default=None, help="Hugging Face user access token.")
    
    args = parser.parse_args()
    
    if args.command == "extract":
        pipeline = CorpusNounPipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            num_workers=args.workers,
            reduce_pure_nouns=args.reduce
        )
        pipeline.run()
    elif args.command == "publish":
        publish_dataset(output_dir=args.output_dir, repo_id=args.repo_id, token=args.token)

if __name__ == "__main__":
    main()
