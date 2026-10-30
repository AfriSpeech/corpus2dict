"""High-throughput multi-process extraction pipeline with checkpoint resumption."""

import os
import glob
import json
import gzip
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import pyarrow as pa
import pyarrow.parquet as pq
import spacy

from corpus2nouns.cleaner import strip_structured_text, chunk_long_text
from corpus2nouns.extractor import extract_noun_phrases
from corpus2nouns.reducer import reduce_to_pure_nouns

def _parse_worker_chunk(texts: list[str]) -> Counter:
    """Worker function executed inside each process in the ProcessPool."""
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    nlp.max_length = 5000000
    counter = Counter()
    for doc in nlp.pipe(texts, batch_size=200):
        counter.update(extract_noun_phrases(doc))
    return counter

def _reduce_worker_chunk(pairs: list[tuple[str, int]]) -> Counter:
    """Worker function for batch adjective reduction."""
    return reduce_to_pure_nouns(pairs)

class CorpusNounPipeline:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        cache_dir: str | None = None,
        num_workers: int = 16,
        reduce_pure_nouns: bool = False
    ):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.cache_dir = os.path.abspath(cache_dir or os.path.join(output_dir, ".cache"))
        self.num_workers = num_workers
        self.reduce_pure_nouns = reduce_pure_nouns
        
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def process_parquet_file(self, fpath: str, subset_name: str) -> Counter:
        """Process a single parquet file, utilizing intermediate .json.gz cache."""
        fname = os.path.basename(fpath)
        cache_subdir = os.path.join(self.cache_dir, subset_name)
        os.makedirs(cache_subdir, exist_ok=True)
        cache_file = os.path.join(cache_subdir, f"{fname}.json.gz")
        
        # Check cache
        if os.path.exists(cache_file):
            try:
                with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                    return Counter(json.load(f))
            except Exception as e:
                print(f"Warning: Corrupt cache {cache_file}, reprocessing: {e}")
                
        # Read from parquet
        t0 = time.time()
        tbl = pq.read_table(fpath)
        texts = []
        
        # Extract text turns (handling messages, conversations, text column, etc.)
        col_names = tbl.column_names
        if "messages" in col_names:
            for msgs in tbl["messages"].to_pylist():
                if isinstance(msgs, list):
                    for m in msgs:
                        content = m.get("content", "") if isinstance(m, dict) else ""
                        cleaned = strip_structured_text(content)
                        if len(cleaned) > 10:
                            for sub_t in chunk_long_text(cleaned):
                                if len(sub_t) > 10:
                                    texts.append(sub_t)
        elif "conversations" in col_names:
            for convs in tbl["conversations"].to_pylist():
                if isinstance(convs, list):
                    for c in convs:
                        val = c.get("value", "") if isinstance(c, dict) else ""
                        cleaned = strip_structured_text(val)
                        if len(cleaned) > 10:
                            for sub_t in chunk_long_text(cleaned):
                                if len(sub_t) > 10:
                                    texts.append(sub_t)
        elif "text" in col_names:
            for txt in tbl["text"].to_pylist():
                cleaned = strip_structured_text(str(txt or ""))
                if len(cleaned) > 10:
                    for sub_t in chunk_long_text(cleaned):
                        if len(sub_t) > 10:
                            texts.append(sub_t)
                            
        if not texts:
            fc = Counter()
        else:
            chunk_size = (len(texts) + self.num_workers - 1) // self.num_workers
            chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
            
            fc = Counter()
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(_parse_worker_chunk, ch) for ch in chunks]
                for fut in futures:
                    fc.update(fut.result())
                    
        # Write to cache
        tmp_cache = cache_file + ".tmp"
        with gzip.open(tmp_cache, "wt", encoding="utf-8") as f:
            json.dump(dict(fc), f)
        os.replace(tmp_cache, cache_file)
        
        el = time.time() - t0
        print(f"  [{subset_name}/{fname}] {len(tbl):,} rows, {len(texts):,} texts -> {sum(fc.values()):,} tokens ({len(fc):,} unique) in {el:.1f}s")
        return fc

    def run(self):
        """Run extraction across all subsets and aggregate global deduped outputs."""
        subsets = sorted([d for d in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, d)) and not d.startswith(".")])
        print(f"Starting CorpusNounPipeline for {len(subsets)} subsets...\n")
        
        t_global_start = time.time()
        subset_counters = {}
        
        for s_idx, subset in enumerate(subsets, 1):
            s_dir = os.path.join(self.input_dir, subset)
            files = sorted(glob.glob(os.path.join(s_dir, "*.parquet")))
            print(f"\n[{s_idx}/{len(subsets)}] Processing {subset} ({len(files)} files)...")
            
            s_counter = Counter()
            for f in files:
                fc = self.process_parquet_file(f, subset)
                s_counter.update(fc)
                
            # If reduce_pure_nouns is requested, drop adjectives/determiners
            if self.reduce_pure_nouns:
                print(f"  Reducing {subset} ({len(s_counter):,} phrases) to pure nouns...")
                pairs = list(s_counter.items())
                chunk_sz = (len(pairs) + self.num_workers - 1) // self.num_workers
                p_chunks = [pairs[i:i+chunk_sz] for i in range(0, len(pairs), chunk_sz)]
                
                reduced_counter = Counter()
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    for fut in [executor.submit(_reduce_worker_chunk, ch) for ch in p_chunks]:
                        reduced_counter.update(fut.result())
                s_counter = reduced_counter
                print(f"  Reduced to {len(s_counter):,} pure nouns.")
                
            subset_counters[subset] = s_counter
            
            # Save subset parquet
            out_sub_dir = os.path.join(self.output_dir, subset)
            os.makedirs(out_sub_dir, exist_ok=True)
            sorted_items = s_counter.most_common()
            sub_rows = [{"phrase": p, "frequency": cnt} for p, cnt in sorted_items]
            sub_schema = pa.schema([
                pa.field("phrase", pa.string()),
                pa.field("frequency", pa.int64())
            ])
            tbl_out = pa.Table.from_pylist(sub_rows, schema=sub_schema)
            out_file = os.path.join(out_sub_dir, "train-00000.parquet")
            pq.write_table(tbl_out, out_file, compression="zstd")
            print(f"  Saved -> {out_file} ({os.path.getsize(out_file)/(1024**2):.2f} MB)")
            
        # Global aggregation
        print("\n========================================================")
        print("AGGREGATING DEDUPED NOUNS ACROSS ALL SUBSETS...")
        print("========================================================")
        global_counter = Counter()
        phrase_subsets = defaultdict(list)
        
        for s_name, counter in subset_counters.items():
            for phrase, count in counter.items():
                global_counter[phrase] += count
                phrase_subsets[phrase].append(s_name)
                
        out_all_dir = os.path.join(self.output_dir, "all")
        os.makedirs(out_all_dir, exist_ok=True)
        
        all_sorted = global_counter.most_common()
        total_unique = len(all_sorted)
        chunk_size = 500000
        num_shards = (total_unique + chunk_size - 1) // chunk_size
        
        all_schema = pa.schema([
            pa.field("phrase", pa.string()),
            pa.field("frequency", pa.int64()),
            pa.field("subset_count", pa.int32()),
            pa.field("subsets", pa.list_(pa.string()))
        ])
        
        for shard_idx in range(num_shards):
            start = shard_idx * chunk_size
            end = min((shard_idx + 1) * chunk_size, total_unique)
            shard_rows = [
                {
                    "phrase": p,
                    "frequency": cnt,
                    "subset_count": len(phrase_subsets[p]),
                    "subsets": phrase_subsets[p]
                }
                for p, cnt in all_sorted[start:end]
            ]
            shard_tbl = pa.Table.from_pylist(shard_rows, schema=all_schema)
            shard_path = os.path.join(out_all_dir, f"train-{shard_idx:05d}-of-{num_shards:05d}.parquet")
            pq.write_table(shard_tbl, shard_path, compression="zstd")
            print(f"  Wrote all shard {shard_idx}: {len(shard_rows):,} rows -> {shard_path}")
            
        print(f"\nPipeline completed in {(time.time()-t_global_start)/60:.1f} minutes!")
        print(f"Total tokens: {sum(global_counter.values()):,}, Unique nouns: {len(global_counter):,}")
