"""
verify_encoder.py - Check if the physics encoder was saved correctly
without BatchNorm layers and can handle single-sample inference.
"""

import torch
import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE, 'src'))

from agent_c_files.physics_modules import PhysicsEncoder, EncoderConfig


def verify_encoder():
    """Verify the encoder checkpoint."""
    
    encoder_path = os.path.join(BASE, 'checkpoints', 'physics_encoder.pt')
    
    print("="*60)
    print("Physics Encoder Verification")
    print("="*60)
    print(f"Encoder path: {encoder_path}")
    print(f"File exists: {os.path.exists(encoder_path)}")
    print()
    
    if not os.path.exists(encoder_path):
        print("❌ ERROR: Encoder file not found!")
        print("Please train the encoder first:")
        print("  python ./src/agent_c_files/train_encoder.py")
        return False
    
    try:
        # Create encoder with BatchNorm disabled
        config = EncoderConfig(
            embedding_dim=16,
            hidden_dim=128,
            num_layers=3,
            dropout=0.1,
            use_batch_norm=False,  # This should match training
        )
        encoder = PhysicsEncoder(config)
        
        # Load checkpoint
        checkpoint = torch.load(encoder_path, map_location='cpu')
        encoder.load_state_dict(checkpoint)
        encoder.eval()
        
        print("✅ Encoder loaded successfully")
        print()
        
        # Check for BatchNorm layers
        has_batchnorm = any(
            isinstance(m, torch.nn.BatchNorm1d) 
            for m in encoder.modules()
        )
        print(f"Encoder has BatchNorm layers: {has_batchnorm}")
        
        if has_batchnorm:
            print("❌ WARNING: Encoder contains BatchNorm layers!")
            print("   This will cause errors during inference with single samples.")
            print("   Retrain with use_batch_norm=False")
        else:
            print("✅ No BatchNorm layers found - Good!")
        print()
        
        # Count layers
        total_layers = len(list(encoder.modules()))
        print(f"Total layers: {total_layers}")
        print(f"Trainable parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        print()
        
        # Test single-sample inference
        print("Testing single-sample inference...")
        params = torch.randn(1, 6)  # Single sample, 6 physics parameters
        
        with torch.no_grad():
            embedding = encoder(params)
        
        print(f"✅ Success! Input shape: {params.shape}")
        print(f"   Output shape: {embedding.shape}")
        print(f"   Embedding: {embedding[0][:5].tolist()}...")
        print()
        
        # Test batch inference
        print("Testing batch inference...")
        batch_params = torch.randn(32, 6)
        with torch.no_grad():
            batch_embedding = encoder(batch_params)
        print(f"✅ Success! Batch shape: {batch_embedding.shape}")
        print()
        
        print("="*60)
        print("✅ VERIFICATION PASSED - Encoder is ready for Agent C!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print()
        print("Possible issues:")
        print("  1. Encoder was trained with BatchNorm (use_batch_norm=True)")
        print("  2. Encoder file is corrupted")
        print("  3. Architecture mismatch")
        print()
        print("Fix: Retrain the encoder with use_batch_norm=False")
        print("  python ./src/agent_c_files/train_encoder.py")
        return False


if __name__ == '__main__':
    success = verify_encoder()
    sys.exit(0 if success else 1)