import os
import logging
from typing import Generator, Dict, Any, Union
import re

# Third-party imports
try:
    from Bio import SeqIO
except ImportError:
    SeqIO = None

try:
    import polars as pl
except ImportError:
    pl = None

# ==========================================
# @Data-Ops: Data Ingestion & Validation
# ==========================================
# Optimizes file I/O for the NTFS USB environment across multiple formats.

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger("@Data-Ops")

def validate_sequence(sequence: str) -> bool:
    """
    Validates that a DNA sequence contains only A, T, C, G characters (case-insensitive).
    Ignores N (ambiguous) or specific gaps for strict validation, or allows them?
    Standard biological validation allows N, but prompt asks to ensure A,T,C,G.
    We will assume strict A,T,C,G + N (often needed) but if prompt says keys, 
    we'll stick to strict or allow N with warning. 
    Strict interpretation: "ensure sequences only contain A, T, C, G".
    """
    if not sequence:
        return False
    
    # Regex for invalid characters
    # ^[ATCGatcg]*$ matches only valid ones.
    return bool(re.match(r'^[ATCGatcg]+$', sequence))

def stream_sequences(file_path: str, chunk_size: int = 1000) -> Generator[Dict[str, Any], None, None]:
    """
    Yields sequences one at a time from FASTA, FASTQ, or PARQUET files.
    Optimized to prevent RAM overflow on limited hardware (32GB USB env).
    
    Args:
        file_path (str): Absolute path to the file.
        chunk_size (int): Rows to process per batch for Parquet files.

    Yields:
        Dict: {"id": str, "sequence": str, "quality": list/None}
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    file_ext = os.path.splitext(file_path)[1].lower()

    # ---------------------------------------------------------
    # Strategy A: BioPython (FASTA/FASTQ)
    # ---------------------------------------------------------
    if file_ext in ['.fasta', '.fa', '.fastq', '.fq']:
        if SeqIO is None:
            logger.error("Biopython not installed. Cannot parse FASTA/FASTQ.")
            return

        file_type = "fastq" if file_ext in ['.fastq', '.fq'] else "fasta"
        
        try:
            with open(file_path, "rt") as handle:
                for record in SeqIO.parse(handle, file_type):
                    seq_str = str(record.seq).upper()
                    
                    if validate_sequence(seq_str):
                        yield {
                            "id": record.id,
                            "sequence": seq_str,
                            "quality": record.letter_annotations.get("phred_quality") if file_type == "fastq" else None
                        }
                    else:
                        logger.warning(f"Skipping Invalid Sequence ID: {record.id} (Contains non-ATCG characters)")

        except Exception as e:
            logger.error(f"Error parsing {file_type} file: {e}")

    # ---------------------------------------------------------
    # Strategy B: Polars (Parquet)
    # ---------------------------------------------------------
    elif file_ext == '.parquet':
        if pl is None:
            logger.error("Polars not installed. Cannot parse Parquet.")
            return

        try:
            # Use LazyFrame scan to avoid loading whole file into RAM
            lf = pl.scan_parquet(file_path)
            
            # Determine total height to manage slicing manually if needed, 
            # or rely on streaming collect capability of Polars (if supported fully for iteration)
            # Safe memory approach: Slice & Collect in Chunks.
            
            # Note: pl.len() is efficient in Lazy execution
            total_rows = lf.select(pl.len()).collect().item()
            logger.info(f"Streaming {total_rows} rows from Parquet: {file_path}")
            
            current_offset = 0
            while current_offset < total_rows:
                # Calculate slice
                current_chunk_size = min(chunk_size, total_rows - current_offset)
                
                # Materialize small chunk
                df_chunk = lf.slice(current_offset, current_chunk_size).collect()
                
                # Iterate rows in chunk
                for row in df_chunk.iter_rows(named=True):
                    # Expecting columns 'id' and 'sequence' or similar standard
                    # Fallbacks for common genomic column names
                    rec_id = row.get('id') or row.get('seq_id') or str(current_offset)
                    seq_str = row.get('sequence') or row.get('seq') or row.get('read')
                    
                    if seq_str and isinstance(seq_str, str):
                        seq_str = seq_str.upper()
                        if validate_sequence(seq_str):
                            yield {
                                "id": str(rec_id),
                                "sequence": seq_str,
                                "quality": row.get('quality')
                            }
                        else:
                            # logging verbosity control to avoid spam
                            pass
                
                current_offset += current_chunk_size
                
        except Exception as e:
            logger.error(f"Error streaming parquet file: {e}")

    else:
        logger.warning(f"Unsupported file format: {file_ext}")

if __name__ == "__main__":
    # Smoke Test for the Parser
    print("Running @Data-Ops Parser Smoke Test...")
    
    # Create a dummy fasta for testing
    test_file = "test_seqs.fasta"
    with open(test_file, "w") as f:
        f.write(">valid_seq\nATCGATCG\n>invalid_seq\nATC GNNN\n")
        
    print(f"\n--- Testing FASTA Stream: {test_file} ---")
    parser = stream_sequences(test_file)
    for seq in parser:
        print(f"Yielded: {seq['id']} | Length: {len(seq['sequence'])}")
        
    # Cleanup
    os.remove(test_file)
    print("\nSmoke Test Complete.")
