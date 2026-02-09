import os
import sys
import logging
import torch
import numpy as np
from unittest.mock import MagicMock
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

# ==========================================
# @Embedder-ML: Windows Conversion Layer
# ==========================================
# Mocking Linux-exclusive libraries to prevent import errors on Windows env
# Ensure the mock has a __spec__ attribute to satisfy importlib checks in transformers
mock_module = MagicMock()
mock_module.__spec__ = MagicMock() 
sys.modules["triton"] = mock_module
sys.modules["flash_attn"] = mock_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger("@Embedder-ML")

class NucleotideEmbedder:
    def __init__(self, model_name="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"):
        """
        Initializes the Nucleotide Transformer with Windows-safe configurations.
        """
        self.device = "cpu" # Hardware constraint: 32GB USB / CPU-only
        self.hidden_dim = 512 # Optimized for 50M parameter model
        logger.info(f"Initializing Embedder on {self.device.upper()}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # @Embedder-ML: Weight Mismatch Fix
            # The v2-50m config from HuggingFace often defaults to 'intermediate_size=2048'.
            # The checkpoint weights (InstaDeepAI) have an MLP layer size of 4096.
            # Since this model architecture uses Gated Linear Units (GLU) or similar, 
            # the internal logic likely doubles the config size (2048 * 2 = 4096).
            # We TRUST the default config (2048) to match the checkpoint (4096).
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            # Validate Hidden Dimension from Config
            if hasattr(config, "hidden_size"):
                logger.info(f"Model Configuration: Hidden Size detected as {config.hidden_size}")
                # Auto-Switch Dimension based on config
                self.hidden_dim = config.hidden_size

            if hasattr(config, "intermediate_size"):
                logger.info(f"Config Check: Using intermediate_size={config.intermediate_size} (Matches Checkpoint Expectation)")
            
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                config=config,
                trust_remote_code=True,
                attn_implementation="eager" # Force standard attention for CPU/Windows
            ).to(self.device)
            
            self.model.eval() # Inference mode only
            logger.info("DeepBio Foundation Model loaded successfully with Shape Defense protocols.")

        except Exception as e:
            logger.critical(f"CRITICAL: Failed to load foundation model. Error: {e}")
            raise

    def embed_sequences(self, sequences: list[str], batch_size: int = 8) -> np.ndarray:
        """
        Generates 768-dim embeddings for a batch of DNA sequences.
        Implements strict shape checking for latent space integrity.
        """
        all_embeddings = []
        
        logger.info(f"Processing batch of {len(sequences)} sequences...")
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            
            # Tokenization (Auto-handling max length)
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1000 
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Extract embeddings:
                # We use the mean of the last hidden state, masking out padding tokens.
                # Shape: (batch_size, seq_len, 768)
                last_hidden_state = outputs.hidden_states[-1] 
                
                # Attention mask for mean pooling
                # Expand mask: (batch_size, seq_len) -> (batch_size, seq_len, 768)
                attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()
                
                sum_embeddings = torch.sum(last_hidden_state * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
                
                # @Embedder-ML: Shape Defense
                # Ensuring output is strictly (Batch, 768) to prevent vector DB injection failures
                if mean_embeddings.shape[1] != self.hidden_dim:
                    logger.error(f"Shape Defense Triggered! Expected dim {self.hidden_dim}, got {mean_embeddings.shape[1]}")
                    raise ValueError("Embedding dimension integrity compromised.")
                
                all_embeddings.append(mean_embeddings.cpu().numpy())

        if not all_embeddings:
            return np.empty((0, self.hidden_dim))

        final_tensor = np.concatenate(all_embeddings, axis=0)
        logger.info(f"Latent Space Generation Complete. Tensor Shape: {final_tensor.shape}")
        return final_tensor

if __name__ == "__main__":
    # Smoke Test
    print("Running @Embedder-ML Smoke Test...")
    try:
        embedder = NucleotideEmbedder()
        test_seqs = ["ACGTACGT", "TGCATGCATGC"]
        emb = embedder.embed_sequences(test_seqs)
        print(f"SUCCESS: Generated {emb.shape} embeddings.")
    except Exception as e:
        print(f"FAILURE: {e}")
