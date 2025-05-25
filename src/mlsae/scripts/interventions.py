# - Pull existing deep SAE from S3
# - Load target LLM
# - Load N rows of dataset, which includes n_ctx tokens (we won't see the logits
# of the last token, but that's fine)
# - For a given feature, retrieve rows that yield nonzero feature activation
# - Compute the control logit for tokens following feature activation
# tokens
#   - Run LLM forward pass until SAE location, then substitute SAE
#   reconstruction, run remainder of forward pass, collect logit for following
#   token
# - Compute the ablated logit for tokens following feature activation tokens
#   - Run LLM forward pass until SAE location, get SAE feature activations, set
#   SAE feature to 0, run decoder, substitute reconstruction, run remainder of
#   forward pass, collect logit for following token
# - Visualization
#   - For a given feature, display a few rows of decoded tokens on which it activates
#   - Highlight tokens (in blue) with magnitude of feature activation on that token
#   - Underline tokens based on the difference between the control logit and the
#   ablated logit. Blue for positive (control - ablated > 0), meaning that the
#   feature increases the probability of that token being produced, and orange
#   for negative, with opacity indicating normalized magnitude (normalized by
#   either that sequence or all sequences)

# %%
# Config
device = "cpu"

# %%
# Load SAE

from mlsae.model import DeepSAE

arch_to_model_id = {
    "0-0": "mildly-good-bear",
    "2-2": "only-suited-cat",
    "2-4-4-2": "merely-finer-feline",
}

# Load SAE from S3
sae = DeepSAE.load(
    "0-0",
    model_id=arch_to_model_id["0-0"],
    load_from_s3=True,
)
sae.to(device)

# %%
# Load target LLM
import transformer_lens

model = transformer_lens.HookedTransformer.from_pretrained(
    "gpt2-small",
).to(device)

# %%
# Load token dataset

from mlsae.data import stream_training_chunks

dataset_iter_batch_size = 100

dataset_iter = iter(stream_training_chunks(
    act_block_size_seqs=dataset_iter_batch_size,
))

# %%
# Load a bunch of tokens

import torch

n_seqs = int(1e2)
n_batches = n_seqs // dataset_iter_batch_size + 1

tokens = []

for i in range(n_batches):
    tokens.append(next(dataset_iter))

tokens = torch.cat(tokens, dim=0)[:n_seqs]

print("Tokens shape:", tokens.shape)

# %%
import numpy as np
from mlsae.config import DTYPES, data_cfg

# Features to collect activations for. This will iterate over raw features in
# the SAE; dead features in this list will not be skipped.
feature_list = np.arange(5)

# (1) We want to run the target LLM on all of the tokens, and store the SAE
# activations for each token
llm_sae_batch_size_seqs = 5
logit_diffs_list = []  # Store differences between control and ablated logits
feature_acts_list = []
mse_list = []

# Process all tokens in batches
with torch.no_grad():
    with torch.autocast(device, dtype=DTYPES[data_cfg.sae_dtype]):
        for start in range(0, tokens.shape[0], llm_sae_batch_size_seqs):
            # Get batch of tokens
            end = min(start + llm_sae_batch_size_seqs, tokens.shape[0])
            token_subblock = tokens[start:end].to(device)
            
            _, cache = model.run_with_cache(
                token_subblock,
                stop_at_layer=data_cfg.layer + 1,
                names_filter=data_cfg.act_name,
            )
            acts = cache.cache_dict[data_cfg.act_name] # [seq tok d_model]
            
            # Store original shape for later
            batch_size, seq_len, d_model = acts.shape
            
            acts_flat = acts.reshape(-1, acts.shape[-1]) # [(seq tok) d_model]

            # Preprocess activations and store so we can apply the inverse
            # transformation on the SAE reconstruction
            acts_mean = acts_flat.mean(dim=-1)
            acts_norm = acts_flat.norm(dim=-1)
            acts_normalized = (acts_flat - acts_mean.unsqueeze(-1)) / acts_norm.unsqueeze(-1)

            _, _, mse, feature_acts, reconstructed = sae.forward(acts_normalized)

            print(f"Batch {start//llm_sae_batch_size_seqs + 1}: MSE = {mse.item():.2e}")

            mse_list.append(mse)
            
            # Reshape arrays back to batch format
            feature_acts_reshaped = feature_acts.reshape(batch_size, seq_len, -1)
            reconstructed_reshaped = reconstructed.reshape(batch_size, seq_len, -1)
            acts_mean_reshaped = acts_mean.reshape(batch_size, seq_len)
            acts_norm_reshaped = acts_norm.reshape(batch_size, seq_len)

            feature_acts_list.append(feature_acts_reshaped[:, :, feature_list])

            # Get ground truth tokens (shifted by 1 for next-token prediction)
            ground_truth_tokens = token_subblock[:, 1:]  # [batch, seq_len-1]
            
            # Initialize tensor to store logit differences for ground truth tokens
            # Shape: (batch, seq_len-1, n_features)
            n_features = len(feature_list)
            logit_diffs = torch.zeros(
                batch_size,
                seq_len - 1,
                n_features,
                device=device
            )

            # For each feature, find where it activates and run targeted forward passes
            for feat_idx, feat_id in enumerate(feature_list):
                # Find all positions where this feature activates
                # feature_acts_reshaped shape: [batch, seq_len, sparse_dim]
                feature_activations = feature_acts_reshaped[:, :, feat_id]  # [batch, seq_len]
                
                # Get positions where feature is non-zero (excluding last position since we need next token)
                activation_positions = torch.nonzero(feature_activations[:, :-1] != 0)  # [n_activations, 2] (batch_idx, seq_idx)
                
                if len(activation_positions) == 0:
                    continue  # Skip if feature doesn't activate
                
                # For each activation position, run a forward pass
                for batch_idx, seq_idx in activation_positions:
                    batch_idx = batch_idx.item()
                    seq_idx = seq_idx.item()
                    
                    # Create control reconstruction (with all features)
                    control_reconstruction = reconstructed_reshaped[batch_idx, seq_idx]
                    control_reconstruction_unnorm = (
                        control_reconstruction * acts_norm_reshaped[batch_idx, seq_idx] + 
                        acts_mean_reshaped[batch_idx, seq_idx]
                    )
                    
                    # Create ablated reconstruction (with this feature zeroed)
                    feature_acts_at_position = feature_acts_reshaped[batch_idx, seq_idx].clone()
                    feature_acts_at_position[feat_id] = 0
                    ablated_reconstruction = sae._decode(feature_acts_at_position.unsqueeze(0)).squeeze(0)
                    ablated_reconstruction_unnorm = (
                        ablated_reconstruction * acts_norm_reshaped[batch_idx, seq_idx] + 
                        acts_mean_reshaped[batch_idx, seq_idx]
                    )
                    
                    # Single hook function that takes the reconstruction as parameter
                    def make_hook_fn(target_batch_idx, target_seq_idx, reconstruction):
                        def hook_fn(acts, hook):
                            acts[target_batch_idx, target_seq_idx] = reconstruction
                            return acts
                        return hook_fn
                    
                    # Get control logits
                    control_hook = make_hook_fn(batch_idx, seq_idx, control_reconstruction_unnorm)
                    logits_control = model.run_with_hooks(
                        token_subblock,
                        fwd_hooks=[(data_cfg.act_name, control_hook)]
                    )
                    
                    # Get ablated logits
                    ablated_hook = make_hook_fn(batch_idx, seq_idx, ablated_reconstruction_unnorm)
                    logits_ablated = model.run_with_hooks(
                        token_subblock,
                        fwd_hooks=[(data_cfg.act_name, ablated_hook)]
                    )
                    
                    # Get the ground truth token for the next position
                    gt_token = ground_truth_tokens[batch_idx, seq_idx]
                    
                    # Extract logits for the ground truth token at the next position
                    control_logit_gt = logits_control[batch_idx, seq_idx, gt_token]
                    ablated_logit_gt = logits_ablated[batch_idx, seq_idx, gt_token]
                    
                    # Store the difference
                    logit_diffs[batch_idx, seq_idx, feat_idx] = control_logit_gt - ablated_logit_gt

            logit_diffs_list.append(logit_diffs)

print(f"\nProcessed {len(tokens)} sequences in {len(logit_diffs_list)} batches")

# %%
# Interactive visualization with navigation
from IPython.display import HTML, display
import json
import time

def create_interactive_feature_visualization(feature_acts_list, logit_diffs_list, tokens, model, max_sequences=10):
    """
    Create an interactive HTML visualization with navigation between features.
    """
    # Generate unique ID for this visualization instance
    viz_id = f"viz_{int(time.time() * 1000)}_{id(feature_acts_list)}"
    
    # Concatenate all results
    all_feature_acts = torch.cat(feature_acts_list, dim=0)  # [total_seqs, seq_len, n_features]
    all_logit_diffs = torch.cat(logit_diffs_list, dim=0)   # [total_seqs, seq_len-1, n_features]
    n_features = all_feature_acts.shape[-1]
    
    # Start building HTML with embedded JavaScript
    html = f"""
    <div id="feature-viz-container-{viz_id}">
        <style>
            #feature-viz-container-{viz_id} {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
            }}
            #feature-viz-container-{viz_id} .navigation {{
                margin: 20px 0;
                text-align: center;
            }}
            #feature-viz-container-{viz_id} .nav-button {{
                padding: 10px 20px;
                margin: 0 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }}
            #feature-viz-container-{viz_id} .nav-button:hover {{
                background-color: #45a049;
            }}
            #feature-viz-container-{viz_id} .nav-button:disabled {{
                background-color: #cccccc;
                cursor: not-allowed;
            }}
            #feature-viz-container-{viz_id} .feature-info {{
                text-align: center;
                margin: 10px 0;
                color: #666;
            }}
            #feature-viz-container-{viz_id} .sequence-row {{ 
                margin: 10px 0; 
                font-family: monospace; 
                font-size: 14px;
                line-height: 1.8;
                white-space: pre-wrap;
                word-wrap: break-word;
                color: black;
            }}
            #feature-viz-container-{viz_id} .token {{ 
                padding: 0px; 
                margin: 0px;
                display: inline-block;
                position: relative;
                color: black;
            }}
            #feature-viz-container-{viz_id} #content-{viz_id} {{
                border: 1px solid #ddd;
                padding: 20px;
                border-radius: 5px;
                margin-top: 20px;
                background-color: white;
            }}
            #feature-viz-container-{viz_id} h2 {{
                color: #333;
            }}
        </style>
        
        <div class="navigation">
            <button class="nav-button" onclick="window['previousFeature_{viz_id}']()">← Previous</button>
            <span class="feature-info">Feature <span id="current-feature-{viz_id}">0</span> of <span id="total-features-{viz_id}">{n_features - 1}</span></span>
            <button class="nav-button" onclick="window['nextFeature_{viz_id}']()">Next →</button>
        </div>
        
        <div id="content-{viz_id}"></div>
        
        <script>
            (function() {{
                let currentFeature = 0;
                const totalFeatures = {n_features};
                
                // Prepare data for all features
                const featureData = {{}};
    """
    
    # Generate visualization data for each feature
    for feature_idx in range(n_features):
        feature_acts_for_feat = all_feature_acts[:, :, feature_idx]
        logit_diffs_for_feat = all_logit_diffs[:, :, feature_idx]
        
        # Normalize by this feature's maximum (per-feature normalization)
        max_feat_act_for_feature = feature_acts_for_feat.abs().max().item()
        max_logit_diff_for_feature = logit_diffs_for_feat.abs().max().item()
        
        normalized_feat_acts = feature_acts_for_feat / max_feat_act_for_feature
        normalized_logit_diffs = logit_diffs_for_feat / max_logit_diff_for_feature
        
        # Find sequences where this feature activates
        sequences_with_activation = torch.any(feature_acts_for_feat != 0, dim=1).nonzero().squeeze(-1)
        
        feature_html = f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
            <h2 style="margin: 0;">Feature {feature_idx}</h2>
            <div style="display: flex; gap: 30px; font-size: 12px; color: #666;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span>Activation:</span>
                    <span class="token" style="background-color: rgba(0, 0, 255, 0.0); padding: 2px 6px;">0</span>
                    <span>→</span>
                    <span class="token" style="background-color: rgba(0, 0, 255, 1.0); padding: 2px 6px;">{max_feat_act_for_feature:.1e}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span>Logit Effect:</span>
                    <span class="token" style="border-bottom: 3px solid rgba(255, 165, 0, 0.0); padding: 2px 6px;">0</span>
                    <span>→</span>
                    <span class="token" style="border-bottom: 3px solid rgba(0, 0, 255, 1.0); padding: 2px 6px;">{max_logit_diff_for_feature:.1e}</span>
                </div>
            </div>
        </div>
        """
        
        if len(sequences_with_activation) == 0:
            feature_html += "<p>No sequences found where this feature activates.</p>"
        else:
            sequences_to_show = sequences_with_activation[:max_sequences]
            
            for seq_idx in sequences_to_show:
                seq_idx = seq_idx.item()
                seq_tokens = tokens[seq_idx]
                
                feature_html += '<div class="sequence-row">'
                
                for token_idx in range(len(seq_tokens)):
                    token_id = seq_tokens[token_idx].item()
                    token_str = model.tokenizer.decode([token_id])
                    
                    # Escape HTML characters first
                    token_str = token_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", "&#39;")
                    
                    # Handle special characters with visual representations
                    # Newlines
                    token_str = token_str.replace('\n', '<span style="color: #888; font-size: 0.8em;">⏎</span>')
                    # Tabs
                    token_str = token_str.replace('\t', '<span style="color: #888; font-size: 0.8em;">⇥</span>')
                    # Carriage returns
                    token_str = token_str.replace('\r', '<span style="color: #888; font-size: 0.8em;">↵</span>')
                    
                    # Convert regular spaces to non-breaking spaces to preserve tokenizer spacing
                    # But first mark any special space sequences
                    token_str = token_str.replace('  ', '<span style="color: #888;">··</span>')  # Double spaces
                    token_str = token_str.replace(' ', '&nbsp;')
                    
                    # Get feature activation for this token
                    feat_act = normalized_feat_acts[seq_idx, token_idx].item()
                    opacity = abs(feat_act) if feat_act != 0 else 0
                    
                    # Determine underline based on logit difference
                    underline_style = ""
                    
                    # Check if we should underline (feature was active on previous token)
                    if token_idx > 0 and feature_acts_for_feat[seq_idx, token_idx - 1] != 0:
                        logit_diff = normalized_logit_diffs[seq_idx, token_idx - 1].item()
                        if logit_diff > 0:
                            underline_opacity = abs(logit_diff)
                            underline_style = f"border-bottom: 3px solid rgba(0, 0, 255, {underline_opacity});"
                        elif logit_diff < 0:
                            underline_opacity = abs(logit_diff)
                            underline_style = f"border-bottom: 3px solid rgba(255, 165, 0, {underline_opacity});"
                    
                    # Build token HTML
                    background_style = f"background-color: rgba(0, 0, 255, {opacity});" if opacity > 0 else ""
                    
                    feature_html += f'<span class="token" style="{background_style} {underline_style}">{token_str}</span>'
                
                feature_html += '</div>'
            
            feature_html += f"<p style='margin-top: 20px; font-size: 12px; color: #666;'>Showing {len(sequences_to_show)} of {len(sequences_with_activation)} sequences where feature {feature_idx} activates.</p>"
        
        # Add to JavaScript object
        html += f"\n                featureData[{feature_idx}] = `{feature_html}`;"
    
    # Complete the HTML with navigation functions
    html += f"""
                
                function updateDisplay() {{
                    document.getElementById('content-{viz_id}').innerHTML = featureData[currentFeature];
                    document.getElementById('current-feature-{viz_id}').textContent = currentFeature;
                    
                    // Update button states
                    const prevButton = document.querySelector('#feature-viz-container-{viz_id} button[onclick="window[\\'previousFeature_{viz_id}\\']()"]');
                    const nextButton = document.querySelector('#feature-viz-container-{viz_id} button[onclick="window[\\'nextFeature_{viz_id}\\']()"]');
                    prevButton.disabled = currentFeature === 0;
                    nextButton.disabled = currentFeature === totalFeatures - 1;
                }}
                
                window['nextFeature_{viz_id}'] = function() {{
                    if (currentFeature < totalFeatures - 1) {{
                        currentFeature++;
                        updateDisplay();
                    }}
                }};
                
                window['previousFeature_{viz_id}'] = function() {{
                    if (currentFeature > 0) {{
                        currentFeature--;
                        updateDisplay();
                    }}
                }};
                
                // Initialize display
                updateDisplay();
            }})();
        </script>
    </div>
    """
    
    return html

# Create and display the interactive visualization
html_output = create_interactive_feature_visualization(
    feature_acts_list, 
    logit_diffs_list, 
    tokens, 
    model,
    max_sequences=10
)
display(HTML(html_output))

# %%

