import os

import torch
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature


class EagleBackbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Eagle-Block2A-2B-v2",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        use_visual_lora: int = 0,
        use_llm_lora: int = 0,
        transformers_loading_kwargs: dict = {},
    ):
        """
        EagleBackbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Eagle-Block2A-2B-v2
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
            use_visual_lora: LoRA rank for visual encoder. 0 = disabled.
            use_llm_lora: LoRA rank for LLM backbone. 0 = disabled.
        """

        super().__init__()

        self._use_visual_lora = use_visual_lora > 0
        self._use_llm_lora = use_llm_lora > 0
        self._visual_lora_rank = use_visual_lora
        self._llm_lora_rank = use_llm_lora

        # Add attention kwargs
        extra_kwargs = {}
        if use_flash_attention:
            extra_kwargs["attn_implementation"] = "flash_attention_2"
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        if model_name == "nvidia/Eagle-Block2A-2B-v2":
            assert use_flash_attention, (
                "nvidia/Eagle-Block2A-2B-v2 requires flash attention by default"
            )
            assert load_bf16, "nvidia/Eagle-Block2A-2B-v2 requires bfloat16 by default"
            eagle_path = os.path.join(os.path.dirname(__file__), "nvidia", "Eagle-Block2A-2B-v2")
            config = AutoConfig.from_pretrained(eagle_path, trust_remote_code=True)
            self.model = AutoModel.from_config(config, trust_remote_code=True)
        else:
            raise ValueError(f"Model {model_name} not supported")

        # needed since we don't use these layers. Also saves compute
        while len(self.model.language_model.model.layers) > select_layer:
            self.model.language_model.model.layers.pop(-1)

        # Apply LoRA after layer truncation so adapters are only on layers in use
        if self._use_visual_lora:
            self.model.wrap_backbone_lora(
                r=use_visual_lora, lora_alpha=2 * use_visual_lora
            )
        if self._use_llm_lora:
            self.model.wrap_llm_lora(
                r=use_llm_lora, lora_alpha=2 * use_llm_lora
            )

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    print(f"Casting trainable parameter {n} to fp32")

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            if self._use_llm_lora:
                for name, p in self.model.language_model.named_parameters():
                    p.requires_grad = "lora_" in name
            else:
                self.model.language_model.requires_grad_(False)
        if not tune_visual:
            if self._use_visual_lora:
                for name, p in self.model.vision_model.named_parameters():
                    p.requires_grad = "lora_" in name
                self.model.mlp1.requires_grad_(False)
            else:
                self.model.vision_model.requires_grad_(False)
                self.model.mlp1.requires_grad_(False)

        if tune_top_llm_layers > 0:
            # Unwrap PEFT model if LoRA was applied (language_model becomes PeftModel)
            llm = self.model.language_model
            if hasattr(llm, 'base_model'):
                llm = llm.base_model.model  # LoraModel -> underlying CausalLM
            for layer in llm.model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        if self._use_visual_lora:
            lora_params = sum(
                p.numel() for n, p in self.model.vision_model.named_parameters() if "lora_" in n
            )
            print(f"Visual encoder LoRA enabled (rank={self._visual_lora_rank}): {lora_params:,} trainable LoRA params")
        if self._use_llm_lora:
            lora_params = sum(
                p.numel() for n, p in self.model.language_model.named_parameters() if "lora_" in n
            )
            print(f"LLM backbone LoRA enabled (rank={self._llm_lora_rank}): {lora_params:,} trainable LoRA params")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Backbone trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
        if trainable == 0:
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        When LoRA is active, keep the module in train mode so LoRA dropout works.
        """
        if self.training:
            if self.model.language_model and not self.tune_llm and not self._use_llm_lora:
                self.model.language_model.eval()
            if self.model.vision_model and not self.tune_visual and not self._use_visual_lora:
                self.model.vision_model.eval()
                self.model.mlp1.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        # 0. Set frozen module to eval
        keys_to_use = ["input_ids", "attention_mask", "pixel_values"]
        vl_input = {k: vl_input[k] for k in keys_to_use}
        outputs = self.model(**vl_input, output_hidden_states=True)
        outputs = outputs["hidden_states"][-1]
        image_mask = vl_input["input_ids"] == self.model.config.image_token_index
        attention_mask = vl_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )  # [B, T2, hidden_size]
