import torch
import torch.nn.functional as F

# Global Cache
_count_cache = {}

def get_count_tensor(H, W, ksize, ssize, psize, device):
    
    key = (H, W, ksize, ssize, psize, str(device))
    
    if key not in _count_cache:
        
        ones_img = torch.ones(1, 1, H + 2*psize, W + 2*psize, device=device)
        
       
        ones_patches = ones_img.unfold(2, ksize, ssize).unfold(3, ksize, ssize)
        H_out, W_out = ones_patches.shape[2], ones_patches.shape[3]
        
        
        ones_flat = ones_patches.permute(0, 1, 4, 5, 2, 3).contiguous()
        ones_flat = ones_flat.view(1, ksize * ksize, H_out * W_out)
        
       
        count = F.fold(ones_flat, (H + 2*psize, W + 2*psize), 
                      kernel_size=ksize, stride=ssize)
        
        _count_cache[key] = count.squeeze(0)  # Rimuovi batch dim, mantieni [1, H_pad, W_pad]
    
    return _count_cache[key]

def svdPatchConv2D_vectorized(I, thresholds, ksize, psize, ssize):
    """
    SVD-based patch denoising con ricostruzione completamente vettorizzata
    
    Args:
        I: Input tensor [B, C, H, W]
        thresholds: Soglie SVD [C, min(ksize, ksize)]
        ksize: Dimensione patch (assumiamo quadrate)
        psize: Padding size
        ssize: Stride size
    
    Returns:
        Tensor denoised [B, C, H, W]
    """
    B, C, H, W = I.shape
    kh, kw = ksize, ksize
    dh, dw = ssize, ssize
    
    # Padding
    imageX = F.pad(I, (psize, psize, psize, psize), mode="constant", value=0)
    
    # Estrazione patch: [B, C, H_out, W_out, kh, kw]
    patches = imageX.unfold(2, kh, dh).unfold(3, kw, dw)
    H_out, W_out = patches.shape[2], patches.shape[3]
    
    # Reshape per batch SVD: [B*C*H_out*W_out, kh, kw]
    patches_reshaped = patches.contiguous().view(-1, kh, kw)
    
    # SVD batch
    U, S, Vh = torch.linalg.svd(patches_reshaped, full_matrices=False)
    
    # Reshape thresholds per batch matching
    N = B * C * H_out * W_out
    thresholds_expanded = thresholds.unsqueeze(0).unsqueeze(2).repeat(B, 1, H_out * W_out, 1)
    thresholds_expanded = thresholds_expanded.view(-1, thresholds.shape[1])  # [N, min(kh, kw)]
    
    # Sogliare i valori singolari (soft thresholding)
    S_thresh = torch.clamp(S - thresholds_expanded, min=0.0)
    
    # Ricostruzione batch patch
    S_diag = torch.diag_embed(S_thresh)  # [N, min(kh, kw), min(kh, kw)]
    patch_rec = U @ S_diag @ Vh  # [N, kh, kw]
    
    # Rimodello in [B, C, H_out, W_out, kh, kw]
    patch_rec = patch_rec.view(B, C, H_out, W_out, kh, kw)
    
    # === RICOSTRUZIONE VETTORIZZATA (NO LOOPS) ===
    
    # Reshape per fold: [B, C*kh*kw, H_out*W_out]
    patches_flat = patch_rec.permute(0, 1, 4, 5, 2, 3).contiguous()
    patches_flat = patches_flat.view(B, C * kh * kw, H_out * W_out)
    
    # Ricostruzione con fold (somma le patch sovrapposte)
    out = F.fold(patches_flat, (H + 2*psize, W + 2*psize), 
                 kernel_size=ksize, stride=ssize)
    
    # Ottieni count tensor pre-calcolato e espandilo per batch e canali
    count = get_count_tensor(H, W, ksize, ssize, psize, I.device)
    count = count.expand(B, C, -1, -1)  # [B, C, H_pad, W_pad]
    
    # Media delle sovrapposizioni
    count = count.clamp(min=1)  # Evita divisione per zero
    out = out / count
    
    # Rimozione padding
    out = out[:, :, psize:psize+H, psize:psize+W]
    
    return out

def clear_count_cache():
    
    global _count_cache
    _count_cache.clear()

# Esempio d'uso
if __name__ == "__main__":
    # Test
    B, C, H, W = 2, 3, 64, 64
    ksize, psize, ssize = 8, 4, 4
    
    I = torch.randn(B, C, H, W)
    thresholds = torch.ones(C, ksize) * 0.1  # Soglie per denoising
    
    result = svdPatchConv2D_vectorized(I, thresholds, ksize, psize, ssize)
    print(f"Input shape: {I.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Cache size: {len(_count_cache)} entries")
