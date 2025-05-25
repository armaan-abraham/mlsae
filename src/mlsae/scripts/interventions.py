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
device = "cuda:0"


# %%
# Load SAE

from mlsae.model import DeepSAE

arch_to_model_id = {
    "0-0": "mildly-good-bear",
    "2-2": "only-suited-cat",
    "2-4-4-2": "merely-finer-feline",
}

# Load SAE from S3
sae_0_0 = DeepSAE.load(
    "0-0",
    model_id=arch_to_model_id["0-0"],
    load_from_s3=True,
)
sae_2_2 = DeepSAE.load(
    "2-2",
    model_id=arch_to_model_id["2-2"],
    load_from_s3=True,
)

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

n_seqs = int(1e3)
n_batches = n_seqs // dataset_iter_batch_size + 1

tokens = []

for i in range(n_batches):
    tokens.append(next(dataset_iter))

tokens = torch.cat(tokens, dim=0)[:n_seqs]

print("Tokens shape:", tokens.shape)

# %%
import numpy as np
from mlsae.config import DTYPES, data_cfg


def collect_feature_activations_and_logit_diffs(
    model,
    sae,
    tokens,
    feature_list,
    batch_size=100,
    device="cuda:0",
):
    """
    Compute SAE feature activations and control-vs-ablated logit differences.

    Returns
    -------
    all_feature_acts : torch.Tensor        # [n_seq, seq_len, n_features]
    all_logit_diffs  : torch.Tensor        # [n_seq, seq_len-1, n_features]
    mse_list         : list[torch.Tensor]  # Per-batch reconstruction MSE
    """
    import torch

    logit_diffs_list = []
    feature_acts_list = []
    mse_list = []

    sae.to(device)

    with torch.no_grad():
        with torch.autocast(device, dtype=DTYPES[data_cfg.sae_dtype]):
            for start in range(0, tokens.shape[0], batch_size):
                # Get batch of tokens
                end = min(start + batch_size, tokens.shape[0])
                token_subblock = tokens[start:end].to(device)

                _, cache = model.run_with_cache(
                    token_subblock,
                    stop_at_layer=data_cfg.layer + 1,
                    names_filter=data_cfg.act_name,
                )
                acts = cache.cache_dict[data_cfg.act_name]  # [batch, seq, d_model]

                # Flatten -> normalise -> SAE forward
                batch_sz, seq_len, d_model = acts.shape
                acts_flat = acts.reshape(-1, d_model)

                acts_mean = acts_flat.mean(dim=-1)
                acts_norm = acts_flat.norm(dim=-1)
                acts_normalized = (
                    acts_flat - acts_mean.unsqueeze(-1)
                ) / acts_norm.unsqueeze(-1)

                _, _, mse, feature_acts, reconstructed = sae.forward(acts_normalized)
                print(
                    f"Batch {start//batch_size + 1}: MSE = {mse.item():.2e}"
                )
                mse_list.append(mse)

                # Reshape back to [batch, seq, …]
                feature_acts_reshaped = feature_acts.reshape(batch_sz, seq_len, -1)
                reconstructed_reshaped = reconstructed.reshape(batch_sz, seq_len, -1)
                acts_mean = acts_mean.reshape(batch_sz, seq_len)
                acts_norm = acts_norm.reshape(batch_sz, seq_len)

                feature_acts_list.append(
                    feature_acts_reshaped[:, :, feature_list]
                )

                # ===== logit differences =====
                n_features = len(feature_list)
                ground_truth_tokens = token_subblock[:, 1:]
                logit_diffs = torch.zeros(
                    batch_sz, seq_len - 1, n_features, device=device
                )

                for feat_idx, feat_id in enumerate(feature_list):
                    feature_activations = feature_acts_reshaped[:, :, feat_id]
                    activation_positions = torch.nonzero(
                        feature_activations[:, :-1] != 0
                    )

                    if len(activation_positions) == 0:
                        continue

                    ctrl_recons, abl_recons = [], []
                    batch_idx_list, seq_idx_list = [], []

                    for b_idx, s_idx in activation_positions:
                        b, s = b_idx.item(), s_idx.item()
                        batch_idx_list.append(b)
                        seq_idx_list.append(s)

                        ctrl_rec = reconstructed_reshaped[b, s]
                        ctrl_rec = (
                            ctrl_rec * acts_norm[b, s] + acts_mean[b, s]
                        )
                        ctrl_recons.append(ctrl_rec)

                        fa_pos = feature_acts_reshaped[b, s].clone()
                        fa_pos[feat_id] = 0
                        abl_rec = sae._decode(fa_pos.unsqueeze(0)).squeeze(0)
                        abl_rec = abl_rec * acts_norm[b, s] + acts_mean[b, s]
                        abl_recons.append(abl_rec)

                    ctrl_recons = torch.stack(ctrl_recons)
                    abl_recons = torch.stack(abl_recons)

                    def make_hook(idxs_b, idxs_s, recons):
                        def hook(tensor, hook=None, **kwargs):
                            for i, (bb, ss) in enumerate(zip(idxs_b, idxs_s)):
                                tensor[bb, ss] = recons[i]
                            return tensor
                        return hook

                    ctrl_logits = model.run_with_hooks(
                        token_subblock,
                        fwd_hooks=[(data_cfg.act_name, make_hook(batch_idx_list, seq_idx_list, ctrl_recons))],
                    )
                    abl_logits = model.run_with_hooks(
                        token_subblock,
                        fwd_hooks=[(data_cfg.act_name, make_hook(batch_idx_list, seq_idx_list, abl_recons))],
                    )

                    for i, (b, s) in enumerate(zip(batch_idx_list, seq_idx_list)):
                        gt = ground_truth_tokens[b, s]
                        logit_diffs[b, s, feat_idx] = (
                            ctrl_logits[b, s, gt] - abl_logits[b, s, gt]
                        )

                logit_diffs_list.append(logit_diffs)

    sae.cpu()

    # -------- concatenate across batches --------
    all_feature_acts = torch.cat(feature_acts_list, dim=0).cpu()
    all_logit_diffs = torch.cat(logit_diffs_list, dim=0).cpu()
    return all_feature_acts, all_logit_diffs, mse_list

# %%

# Features to collect activations for. This will iterate over raw features in
# the SAE; dead features in this list will not be skipped.
feature_list = np.arange(100)

# Collect activations / logit diffs in one call
all_feature_acts_0_0, all_logit_diffs_0_0, mse_list_0_0 = collect_feature_activations_and_logit_diffs(
    model=model,
    sae=sae_0_0,
    tokens=tokens,
    feature_list=feature_list,
    device=device,
)

print(f"\nCollected activations for {all_feature_acts_0_0.shape[0]} sequences")

# %%

# Collect activations / logit diffs in one call
all_feature_acts_2_2, all_logit_diffs_2_2, mse_list_2_2 = collect_feature_activations_and_logit_diffs(
    model=model,
    sae=sae_2_2,
    tokens=tokens,
    feature_list=feature_list,
    device=device,
)

print(f"\nCollected activations for {all_feature_acts_0_0.shape[0]} sequences")

# %%
# Interactive visualization with navigation
from IPython.display import HTML, display
import json
import time
import html
import unicodedata

def create_interactive_feature_visualization(
    all_feature_acts,
    all_logit_diffs,
    tokens,
    model,
    max_sequences=10,
    title: str | None = None,
    start_feature: int = 0,
):
    """
    Create an interactive HTML visualization with navigation between features.
    Expects *concatenated* tensors for feature activations and logit diffs.

    Parameters
    ----------
    start_feature : int, optional
        Which feature index to display first (default 0).
    """
    # Generate unique ID for this visualization instance
    viz_id = f"viz_{int(time.time() * 1000)}_{id(all_feature_acts)}"

    n_features = all_feature_acts.shape[-1]
    
    # Ensure the starting feature is within valid range
    start_feature = int(max(0, min(start_feature, n_features - 1)))
    
    # Optional title element
    title_html = f"<h2 class='viz-title'>{title}</h2>" if title else ""
    
    # Start building HTML with embedded JavaScript
    html_content = f"""
    <div id="feature-viz-container-{viz_id}">
        <meta charset="UTF-8">
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
                font-size: 12px;
                line-height: 1.3;
                white-space: pre-wrap;
                word-wrap: break-word;
                color: black;
            }}
            #feature-viz-container-{viz_id} .token-span {{ 
                padding: 3px 0px;
                margin: 0;
                display: inline;
                position: relative;
                color: black;
                background-clip: content-box;
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
            #feature-viz-container-{viz_id} .viz-title {{
                text-align: center;
                margin: 10px 0 5px 0;
                color: #222;
            }}
        </style>
        
        {title_html}
        <div class="navigation">
            <button class="nav-button" onclick="window['previousFeature_{viz_id}']()">← Previous</button>
            <span class="feature-info">Feature <span id="current-feature-{viz_id}">0</span> of <span id="total-features-{viz_id}">{n_features - 1}</span></span>
            <button class="nav-button" onclick="window['nextFeature_{viz_id}']()">Next →</button>
        </div>
        
        <div id="content-{viz_id}"></div>
        
        <script>
            (function() {{
                let currentFeature = {start_feature};
                const totalFeatures = {n_features};
                
                // Prepare data for all features
                const featureData = {{}};
    """
    
    def get_token_char_spans(token_ids, tokenizer):
        """
        Get character spans for each token in the decoded text.
        Returns list of (start_char, end_char) tuples.
        """
        char_spans = []
        for i in range(len(token_ids)):
            prefix_text = tokenizer.decode(token_ids[:i]) if i > 0 else ""
            current_text = tokenizer.decode(token_ids[:i+1])
            start_char = len(prefix_text)
            end_char = len(current_text)
            char_spans.append((start_char, end_char))
        return char_spans
    
    def escape_html_preserve_structure(text):
        """
        Escape HTML characters while preserving whitespace structure.
        """
        # Basic HTML escaping
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        text = text.replace('"', '&quot;').replace("'", "&#39;")
        
        # Handle special whitespace characters with visual representations
        text = text.replace('\n', '<span style="color: #888; font-size: 0.8em;">⏎</span>')
        text = text.replace('\t', '<span style="color: #888; font-size: 0.8em;">⇥</span>')
        text = text.replace('\r', '<span style="color: #888; font-size: 0.8em;">↵</span>')
        
        # Convert spaces to non-breaking spaces to preserve tokenizer spacing
        text = text.replace('  ', '<span style="color: #888;">··</span>')  # Double spaces
        text = text.replace(' ', '&nbsp;')
        
        return text
    
    # Generate visualization data for each feature
    for feature_idx in range(n_features):
        feature_acts_for_feat = all_feature_acts[:, :, feature_idx]
        logit_diffs_for_feat = all_logit_diffs[:, :, feature_idx]
        
        # Normalize by this feature's maximum (per-feature normalization)
        max_feat_act_for_feature = feature_acts_for_feat.abs().max().item()
        max_logit_diff_for_feature = logit_diffs_for_feat.abs().max().item()
        
        if max_feat_act_for_feature == 0:
            max_feat_act_for_feature = 1.0  # Avoid division by zero
        if max_logit_diff_for_feature == 0:
            max_logit_diff_for_feature = 1.0
        
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
                    <span class="token-span" style="background-color: rgba(0, 0, 255, 0.0); padding: 2px 6px;">0</span>
                    <span>→</span>
                    <span class="token-span" style="background-color: rgba(0, 0, 255, 1.0); padding: 2px 6px;">{max_feat_act_for_feature:.1e}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span>Logit Effect:</span>
                    <span class="token-span" style="border-bottom: 3px solid rgba(255, 165, 0, 0.0); padding: 2px 6px;">0</span>
                    <span>→</span>
                    <span class="token-span" style="border-bottom: 3px solid rgba(0, 0, 255, 1.0); padding: 2px 6px;">{max_logit_diff_for_feature:.1e}</span>
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
                
                # Decode the entire sequence at once
                full_text = model.tokenizer.decode(seq_tokens)
                
                # Get character spans for each token
                char_spans = get_token_char_spans(seq_tokens, model.tokenizer)
                
                # Escape the full text for HTML
                escaped_text = escape_html_preserve_structure(full_text)
                
                feature_html += '<div class="sequence-row">'
                
                # Build HTML with proper token highlighting
                current_pos = 0
                
                for token_idx in range(len(seq_tokens)):
                    start_char, end_char = char_spans[token_idx]
                    
                    # Add any text between previous token and current token (shouldn't happen but safety)
                    if start_char > current_pos:
                        between_text = escaped_text[current_pos:start_char]
                        feature_html += between_text
                    
                    # Get the text for this token
                    token_text = escaped_text[start_char:end_char]
                    
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
                    
                    feature_html += f'<span class="token-span" style="{background_style} {underline_style}">{token_text}</span>'
                    
                    current_pos = end_char
                
                # Add any remaining text (shouldn't happen but safety)
                if current_pos < len(escaped_text):
                    feature_html += escaped_text[current_pos:]
                
                feature_html += '</div>'
            
            feature_html += f"<p style='margin-top: 20px; font-size: 12px; color: #666;'>Showing {len(sequences_to_show)} of {len(sequences_with_activation)} sequences where feature {feature_idx} activates.</p>"
        
        # Add to JavaScript object (escape for JavaScript string)
        feature_html_escaped = feature_html.replace('`', '\\`').replace('\\', '\\\\').replace('${', '\\${')
        html_content += f"\n                featureData[{feature_idx}] = `{feature_html_escaped}`;"
    
    # Complete the HTML with navigation functions
    html_content += f"""
                
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
    
    return html_content

# Create and display the interactive visualization
html_output = create_interactive_feature_visualization(
    all_feature_acts_0_0, 
    all_logit_diffs_0_0, 
    tokens, 
    model,
    max_sequences=10,
    title="Shallow SAE",
    start_feature=23,
)
display(HTML(html_output))

# %%

