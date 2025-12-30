import unittest
from unittest.mock import patch, MagicMock
import importlib.util
import sys
import os

# 1. DYNAMIC IMPORT BOILERPLATE
# This allows us to import 'ollama-ops.py' (hyphen) as module 'ollama_ops' (underscore)
# -------------------------------------------------------------------------------------
# Ensure we are looking in the current directory
script_path = os.path.join(os.getcwd(), "ollama-ops.py")

# Create the spec using the actual filename
spec = importlib.util.spec_from_file_location("ollama_ops", script_path)
ollama_ops = importlib.util.module_from_spec(spec)

# Register it in sys.modules so @patch decorators can find "ollama_ops"
sys.modules["ollama_ops"] = ollama_ops

# Execute the module
spec.loader.exec_module(ollama_ops)
# -------------------------------------------------------------------------------------

class TestOllamaOps(unittest.TestCase):

    # --- 1. Test Pure Logic ---

    def test_parse_size_input(self):
        # Tokens
        self.assertEqual(ollama_ops.parse_size_input("32k"), (32768, "32"))
        self.assertEqual(ollama_ops.parse_size_input("8192"), (8388608, "8192")) 
        
        # Bytes
        self.assertEqual(ollama_ops.parse_size_input("1048576b"), (1048576, "1024"))
        
        # Invalid
        self.assertEqual(ollama_ops.parse_size_input("invalid"), (None, None))

    def test_estimate_architecture(self):
        # 8B model (Llama 3 size)
        layers, _, _, _ = ollama_ops.estimate_architecture_from_params(8.0)
        self.assertEqual(layers, 32)
        
        # 70B model
        layers, _, _, _ = ollama_ops.estimate_architecture_from_params(70.0)
        self.assertEqual(layers, 80)
        
        # Tiny model
        layers, _, _, _ = ollama_ops.estimate_architecture_from_params(1.8)
        self.assertEqual(layers, 32) # Fallback for <4B

    # --- 2. Test Parsers (Mocking CLI Output) ---

    @patch('shutil.which')
    @patch('subprocess.run')
    def test_scrape_cli_metadata(self, mock_run, mock_which):
        # Simulate ollama installed
        mock_which.return_value = "/usr/bin/ollama"
        
        # Simulate "ollama show" output
        mock_stdout = """
          Model
            architecture        llama
            parameters          8.0B
            quantization        Q4_0
            context length      8192
            embedding length    4096
        """
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_stdout)

        data = ollama_ops.scrape_cli_metadata("llama3")
        
        self.assertEqual(data.get('parameters_b'), 8.0)
        self.assertEqual(data.get('context_length'), 8192)
        self.assertEqual(data.get('embedding_length'), 4096)

    # --- 3. Test GPU Logic (Mocking nvidia-smi) ---

    # NOTE: The patch string MUST use "ollama_ops" because we registered it in sys.modules
    @patch('ollama_ops._require_nvidia')
    @patch('subprocess.check_output')
    def test_get_gpu_free_memory(self, mock_sub, mock_check):
        mock_check.return_value = True # Simulate NVIDIA present
        # Simulate nvidia-smi returning "24000" (MB)
        mock_sub.return_value = "24000" 
        
        bytes_free = ollama_ops.get_gpu_free_memory()
        
        expected_bytes = 24000 * 1024 * 1024
        self.assertEqual(bytes_free, expected_bytes)

    # --- 4. Test Optimization Logic ---
    
    @patch('builtins.input', return_value='y') 
    @patch('ollama_ops.ollama.generate') 
    @patch('ollama_ops.get_gpu_free_memory')
    @patch('ollama_ops.ollama.show')
    @patch('ollama_ops.ollama.list')
    @patch('ollama_ops.create_context_variant')
    def test_calculate_max_context_logic(self, mock_create, mock_list, mock_show, mock_gpu, mock_generate, mock_input):
        """
        Scenario: Standard Llama 3 optimization
        """
        mock_gpu.return_value = 12 * 1024**3
        
        mock_model_obj = MagicMock()
        mock_model_obj.model = "llama3:latest"
        mock_model_obj.size = 4 * 1024**3
        mock_list.return_value.models = [mock_model_obj]

        mock_show.return_value.model_info = {
            "general.parameter_count": 8000000000,
            "llama.block_count": 32,
            "llama.embedding_length": 4096,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8, 
            "llama.context_length": 128000 
        }

        existing = set()
        ollama_ops.calculate_max_context("llama3:latest", existing, buffer_gb=1.0, batch_mode=False)

        self.assertTrue(mock_create.called)
        args, _ = mock_create.call_args
        model_name, ctx_bytes, ctx_label = args
        self.assertEqual(model_name, "llama3:latest")
        self.assertTrue(int(ctx_label) > 16) 

    # --- 5. Test Filters (Cloud/Size) ---

    @patch('ollama_ops.ollama.generate')
    @patch('ollama_ops.get_gpu_free_memory')
    @patch('ollama_ops.ollama.list')
    @patch('ollama_ops.create_context_variant')
    def test_filters_cloud_and_size(self, mock_create, mock_list, mock_gpu, mock_generate):
        """
        Scenario: 
        1. 'qwen:cloud' should be skipped by name.
        2. 'tiny-stub' should be skipped by size (<200MB).
        """
        mock_gpu.return_value = 24 * 1024**3 # Lots of RAM
        
        # Setup Inventory
        # Model A: Cloud name
        model_cloud = MagicMock()
        model_cloud.model = "qwen:cloud"
        model_cloud.size = 5 * 1024**3 # Normal size, but bad name
        
        # Model B: Tiny size
        model_tiny = MagicMock()
        model_tiny.model = "tiny-stub"
        model_tiny.size = 100 * 1024 # 100KB (Too small)

        mock_list.return_value.models = [model_cloud, model_tiny]

        existing = set()

        # Test 1: Cloud Name Filter
        ollama_ops.calculate_max_context("qwen:cloud", existing, batch_mode=True)
        # Should NOT call create (skipped immediately)
        mock_create.assert_not_called()

        # Test 2: Small Size Filter
        ollama_ops.calculate_max_context("tiny-stub", existing, batch_mode=True)
        # Should NOT call create (skipped after size check)
        mock_create.assert_not_called()

    # --- 6. Test Batch Deletion ---

    @patch('ollama_ops.ollama.delete')
    @patch('builtins.input')
    @patch('ollama_ops.get_available_models')
    def test_delete_interactive(self, mock_get_models, mock_input, mock_delete):
        """
        Scenario: 
        1. Mock inventory with 3 models.
        2. Test 'y' confirmation deletes matches.
        3. Test 'n' aborts.
        """
        # Setup Inventory
        m1 = MagicMock(); m1.model = "keep-me"
        m2 = MagicMock(); m2.model = "delete-me-ctx32k"
        m3 = MagicMock(); m3.model = "delete-me-too-ctx16k"
        
        mock_get_models.return_value = [m1, m2, m3]

        # Case A: User Confirms Yes
        mock_input.return_value = 'y'
        ollama_ops.delete_models_interactive("delete*", skip_confirm=False)
        
        # Expect 2 deletes
        self.assertEqual(mock_delete.call_count, 2)
        mock_delete.assert_any_call("delete-me-ctx32k")
        mock_delete.assert_any_call("delete-me-too-ctx16k")
        
        # Reset
        mock_delete.reset_mock()

        # Case B: User Aborts
        mock_input.return_value = 'n'
        ollama_ops.delete_models_interactive("delete*", skip_confirm=False)
        mock_delete.assert_not_called()

        # Case C: Force Flag (No Input)
        mock_input.reset_mock()
        ollama_ops.delete_models_interactive("delete*", skip_confirm=True)
        self.assertEqual(mock_delete.call_count, 2)
        mock_input.assert_not_called()

if __name__ == '__main__':
    unittest.main()