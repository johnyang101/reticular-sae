import torch
from transformers import EsmForProteinFolding

class Hooked_ESMFold_Ablation(EsmForProteinFolding):
    def __init__(self, config, hidden_state_to_keep_idx=36):
        """
        Custom ESMFold model with the ability to ablate specific hidden states.
        
        This class extends EsmForProteinFolding to allow selective ablation of hidden states
        from the ESM-2 language model component. By default, it keeps only a single specified
        hidden state (layer 36 by default) and zeros out all others.
        
        Args:
            config: AutoConfig class from huggingface transformers.
            hidden_state_to_keep_idx: Index of the hidden state to preserve during ablation.
                                      If None, no ablation is performed.
                                      Note: Hidden states are indexed as follows:
                                      - 0: embedding layer output
                                      - 1 to N: transformer layer outputs
                                      - N+1: final output layer
                                      Default: 36 (corresponds to the 36th transformer layer)
                                      
        Example:
            ```python
            from transformers import AutoConfig
            
            # Load the ESMFold configuration
            config = AutoConfig.from_pretrained("facebook/esmfold_v1")
            
            # Create the ablated model keeping only layer 36
            abl_esmf_model = ESMFold_Custom_ESM2_Ablation(config=config, hidden_state_to_keep_idx=36)
            
            # Load the pretrained weights
            abl_esmf_model.load_state_dict(
                EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=False).state_dict()
            )
            
            # Move model to GPU if available
            abl_esmf_model = abl_esmf_model.cuda()
            
            # Now you can use the model for inference with only layer 36 active
            # For example, to predict the structure of a protein sequence:
            # outputs = abl_esmf_model(amino_acid_indices)
            ```
        """
        super().__init__(config)
        self.hidden_state_to_keep_idx = hidden_state_to_keep_idx

    def compute_language_model_representations(self, esmaa: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        B, L = esmaa.shape  # B = batch size, L = sequence length.

        if self.config.esmfold_config.bypass_lm:
            esm_s = torch.zeros(B, L, self.esm_s_combine.size[0], -1, self.esm_feats, device=device)
            return esm_s

        bosi, eosi = self.esm_dict_cls_idx, self.esm_dict_eos_idx
        bos = esmaa.new_full((B, 1), bosi)
        eos = esmaa.new_full((B, 1), self.esm_dict_padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(B), (esmaa != 1).sum(1)] = eosi

        # _, esm_z, esm_s = self.esm(esmaa, return_pairs=self.config.esmfold_config.use_esm_attn_map)
        # Because we do not support use_esm_attn_map in the HF port as it is not used in any public models,
        # esm_z is always None
        esm_hidden_states = self.esm(esmaa, attention_mask=esmaa != 1, output_hidden_states=True)["hidden_states"]
        esm_s = torch.stack(esm_hidden_states, dim=2)

        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C

        #Ablation Code
        if self.hidden_state_to_keep_idx is not None:
            esm_s_ablated = torch.zeros_like(esm_s)
            #NOTE: Hidden state is num_layers + 1, 0 idx is embedding, last idx is output, should use 1-idxed layer idx to idx hidden state properly
            esm_s_ablated[:, :, self.hidden_state_to_keep_idx, :] = esm_s[:, :, self.hidden_state_to_keep_idx, :]
            return esm_s_ablated

        return esm_s

