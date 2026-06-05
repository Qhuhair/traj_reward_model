import os
import yaml
from adapters.deepseek_adapters import OpenAIStyleAdapter
from adapters.qwen_vllm_adapter import QwenVLLMAdapter
from adapters.ollama_adapter import OllamaAdapter
from adapters.remote_qwen_adapter import RemoteQwenAdapter
from utils import ResponseParser, DebugLogger

class LLMCaller:
    def __init__(self, config_path="config.yaml", model: str = None, prompt: str = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(config_path):
            config_path = os.path.join(base_dir, config_path)

        # 1. 加载基础配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        active_model = model or self.config['active_model']
        active_prompt = prompt or self.config['active_prompt']
        model_cfg = self.config['models'][active_model]
        
        # 2. 适配器工厂
        adapter_map = {
            "openai_style": OpenAIStyleAdapter,
            "qwen_vllm": QwenVLLMAdapter,
            "ollama": OllamaAdapter,
            "remote_qwen": RemoteQwenAdapter,
        }
        adapter_class = adapter_map.get(model_cfg['protocol'], OpenAIStyleAdapter)
        self.adapter = adapter_class(model_cfg)

        # 3. 加载指定的 Prompt 模板
        prompt_file = os.path.join(base_dir, "prompts", f"{active_prompt}.yaml")
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.prompt_template = yaml.safe_load(f)['template']
            
        self.logger = DebugLogger(enabled=self.config.get('debug', True))

    def call(self, **kwargs):
        """
        使用关键字参数动态填充 Prompt。
        image_before / image_after 会传递给多模态 adapter。
        """
        full_prompt = self.prompt_template.format(**kwargs)
        
        import time
        start_time = time.time()
        raw_response = self.adapter.invoke(
            full_prompt,
            image_before=kwargs.get("image_before"),
            image_after=kwargs.get("image_after"),
        )
        duration = time.time() - start_time
        
        self.logger.log_call(
            self.config.get('active_model', 'unknown'),
            full_prompt, raw_response, duration
        )
        return ResponseParser.parse(raw_response)