import torch
import psutil
import cpuinfo


class TorchDeviceManager:
    def __init__(self, preferred_device=None):
        self.xpu_module_available = self._is_module_available('torch.xpu')
        self.ipex_module_available = self._is_module_available('intel_extension_for_pytorch')
        self.mps_module_available = self._is_module_available('torch.backends.mps')
        self.cuda_module_available = self._is_module_available('torch.cuda')
        self.valid_devices = self._determine_valid_devices()
        self.device = self._initialize_device(preferred_device)

    def _initialize_device(self, preferred_device):
        if preferred_device and preferred_device not in self.valid_devices:
            raise ValueError(f"Preferred device '{preferred_device}' is invalid.")
        if not preferred_device:
            return self.valid_devices[0]
        return preferred_device

    def _is_module_available(self, module_name):
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def _determine_valid_devices(self):
        valid_devices = []
        if self.cuda_module_available and torch.cuda.is_available():
            valid_devices.append('cuda')
            for i in range(torch.cuda.device_count()):
                valid_devices.append(f'cuda:{i}')
        if self.xpu_module_available and self.ipex_module_available and self._is_device_valid('xpu'):
            valid_devices.append('xpu')
            for i in range(torch.xpu.device_count()):
                valid_devices.append(f'xpu:{i}')
        if self.mps_module_available and torch.backends.mps.is_available():
            valid_devices.append('mps')
        valid_devices.append('cpu')
        return valid_devices

    def _is_device_valid(self, device_name):
        try:
            torch.tensor([1.0], device=device_name)
            return True
        except:
            return False

    def list_devices(self):
        for valid_device in self.valid_devices:
            if 'xpu' in valid_device:
                print(f'[{valid_device}]: {torch.xpu.get_device_properties(valid_device)}')
            elif 'cuda' in valid_device:
                print(f'[{valid_device}]: {torch.cuda.get_device_properties(valid_device)}')
            elif valid_device == 'mps':
                print(f'[{valid_device}]: On-board device')
            elif valid_device == 'cpu':
                total_memory_mb = int(psutil.virtual_memory().total / (1024 * 1024))
                cpu_name = str(cpuinfo.get_cpu_info()['brand_raw'])
                num_cpu_threads = torch.get_num_threads()
                num_interop_threads = torch.get_num_interop_threads()
                print(f'[{valid_device}]: name=\'{cpu_name}\', total_memory={total_memory_mb}MB, torch_threads={num_cpu_threads}, torch_interop_threads={num_interop_threads}')

    def stage_model(self, model):
        if self.using_gpu():
            model = model.to(self.device)
        if self.device in ['cpu', 'xpu'] and self.ipex_module_available:
            import intel_extension_for_pytorch as ipex
            model = ipex.optimize(model)
        return model

    def stage_data(self, data):
        if self.using_gpu():
            data = data.to(self.device)
        return data

    def using_gpu(self):
        return self.device != 'cpu'

    def using_cpu(self):
        return self.device == 'cpu'
