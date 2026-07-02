import json
import torch
import triton_python_backend_utils as pb_utils
from triton_kernels import fast_nudft_reconstruct

class TritonPythonModel:
    def initialize(self, args):
        """
        Loads the learned frequencies and coefficients into GPU memory once.
        In production, this reads from an artifact store (S3/local).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mocking the loaded parameters from the `fit()` stage
        # K = 1024 frequencies
        K = 1024
        self.f = torch.linspace(0, 0.5, K, device=self.device)
        self.F = torch.complex(
            torch.randn(K, device=self.device), 
            torch.randn(K, device=self.device)
        )
        
        # JIT compile the Triton kernel on startup by running a dummy payload
        _ = fast_nudft_reconstruct(
            torch.zeros(128, device=self.device), self.f, self.F
        )

    def execute(self, requests):
        """
        Processes batches of requests zero-copy via DLPack.
        """
        responses = []
        for request in requests:
            # 1. Zero-copy transfer from network interface to PyTorch GPU tensor
            in_tensor = pb_utils.get_input_tensor_by_name(request, "TIMESTAMPS")
            t_pt = torch.from_dlpack(in_tensor.to_dlpack()).to(self.device)
            
            # 2. Execute OpenAI Triton kernel
            out_pt = fast_nudft_reconstruct(t_pt, self.f, self.F)
            
            # 3. Zero-copy transfer back to the network response
            out_tensor = pb_utils.Tensor.from_dlpack("RECONSTRUCTED_SIGNAL", torch.to_dlpack(out_pt))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            
        return responses