import numpy as np

def policy_action(loaded, observation):
    """3-hidden-layer network (256-256-128)"""
    # Parameter loading logic
    if isinstance(loaded, (str, bytes)):
        params = np.load(loaded, allow_pickle=True)
    else:
        params = loaded
    
    # Verify parameter size for deep network
    expected_size = 101508  # (8*256 +256) + (256*256 +256) + (256*128 +128) + (128*4 +4)
    if params.size != expected_size:
        raise ValueError(f"Expected {expected_size} params, got {params.size}")
    
    # Parameter unpacking
    idx = 0
    # Layer 1: 8->256
    W1 = params[idx:idx+2048].reshape(256, 8); idx += 2048
    b1 = params[idx:idx+256]; idx += 256
    # Layer 2: 256->256
    W2 = params[idx:idx+65536].reshape(256, 256); idx += 65536
    b2 = params[idx:idx+256]; idx += 256
    # Layer 3: 256->128 
    W3 = params[idx:idx+32768].reshape(128, 256); idx += 32768
    b3 = params[idx:idx+128]; idx += 128
    # Action layer: 128->4
    W4 = params[idx:idx+512].reshape(4, 128); idx += 512
    b4 = params[idx:idx+4]; idx += 4

    # Forward pass
    hidden = np.maximum(observation @ W1.T + b1, 0)
    hidden = np.maximum(hidden @ W2.T + b2, 0)
    hidden = np.maximum(hidden @ W3.T + b3, 0)
    logits = hidden @ W4.T + b4
    
    return int(np.argmax(logits))