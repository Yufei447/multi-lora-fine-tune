from aspen.modelargs import LLMModelArgs, MultiLoraBatchData
from aspen.checkpoint import CheckpointRecomputeFunction
from aspen.model import repeat_kv, apply_rotary_emb, precompute_rope_angle, precompute_mask
from aspen.model import LLMModel, RMSNorm
from aspen.LoraLiner import Linear

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import xformers.ops
import xformers.ops.fmha.attn_bias
from transformers import LlamaForCausalLM
from typing import List, Dict, Tuple


class Transformer():
    def __init__(self, layer_id: int, args: LLMModelArgs):
        # attention
        self.wq_: Linear = None  # dim * dim
        self.wk_: Linear = None  # dim * dim
        self.wv_: Linear = None  # dim * dim
        self.wo_: Linear = None  # dim * dim
        # feed forward
        self.w1_: Linear = None  # also gate FNN * dim
        self.w2_: Linear = None  # also down dim * FNN
        self.w3_: Linear = None  # also up   FNN * dim
        # norm
        self.attention_norm_: RMSNorm = None  # dim
        self.ffn_norm_: RMSNorm = None        # dim
        # other arg
        self.layer_id_ = layer_id
        self.norm_eps_ = args.norm_eps_
        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.n_rep = self.n_heads_ // self.n_kv_heads_
        self.head_dim_ = args.dim_ // args.n_heads_

    def init_lora_layer_weight(self,
                               adapter_name: str,
                               r: int,
                               lora_alpha: int,
                               lora_dropout: float,
                               target: Dict[str, bool]):
        linear_layer_list = [self.wk_, self.wq_, self.wv_,
                             self.wo_, self.w1_, self.w2_, self.w3_]
        linear_layer_name_list = [
            "k_proj", "q_proj", "v_proj", "o_proj", "w1_proj", "w2_proj", "w3_proj"]

        for idx, layer_name in enumerate(linear_layer_name_list):
            if layer_name in target and target[layer_name]:
                linear_layer_list[idx].init_lora_weight(
                    adapter_name, r, lora_alpha, lora_dropout)

    # @torch.compile
    def forward(self,
                data: torch.Tensor,
                mask: torch.Tensor,
                rope_angle: Tuple[torch.Tensor, torch.Tensor],
                input_args: MultiLoraBatchData):
        batch_size, max_seq_len, _ = data.shape

        attention_norm_data = self.attention_norm_.forward(data)

        xq = self.wq_.forward(attention_norm_data, input_args)
        xk = self.wk_.forward(attention_norm_data, input_args)
        xv = self.wv_.forward(attention_norm_data, input_args)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_, self.head_dim_)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_)

        # apply rotary embedding
        xq, xk = apply_rotary_emb(xq, xk, rope_angle)

        # inference model use cache-kv
        if input_args.inference_model_:
            if len(input_args.cache_key_) <= self.layer_id_:
                input_args.cache_key_.append(xk)
                input_args.cache_value_.append(xv)
            else:
                xk = torch.cat(
                    (input_args.cache_key_[self.layer_id_], xk), dim=1)
                xv = torch.cat(
                    (input_args.cache_value_[self.layer_id_], xv), dim=1)
                input_args.cache_key_[self.layer_id_] = xk
                input_args.cache_value_[self.layer_id_] = xv

        # for llama2 need to repeat the heads
        # before dim: batch_size, seq_len, n_kv_head, head_dim
        # after dim: batch_size, seq_len, n_head, head_dim
        xq = repeat_kv(xq, self.n_rep)
        xv = repeat_kv(xq, self.n_rep)

        if input_args.inference_model_:
            attention_score = xformers.ops.memory_efficient_attention(
                xq, xk, xv, attn_bias=xformers.ops.fmha.attn_bias.LowerTriangularMask()
            )
        else:
            attention_score = xformers.ops.memory_efficient_attention(
                xq, xk, xv, mask)
        attention_score = attention_score.view(batch_size, max_seq_len, -1)

        # get output attention score
        data = data + self.wo_.forward(attention_score, input_args)

        # feed forward fully connected
        score_norm_data = self.ffn_norm_.forward(data)
        w1 = self.w1_.forward(score_norm_data, input_args)
        w3 = self.w3_.forward(score_norm_data, input_args)

        data = data + self.w2_.forward(F.silu(w1) * w3, input_args)

        return data


class LlamaModel(LLMModel):
    def __init__(self, args: LLMModelArgs):
        # weight
        self.token_embedding_: torch.Tensor = None

        self.layers_: List[Transformer] = []
        for layer_id in range(args.n_layers_):
            self.layers_.append(Transformer(layer_id, args))

        self.norm_: RMSNorm = None          # dim
        self.output_: torch.Tensor = None   # vocab size * dim

        # cos and sin
        self.rope_angle_: Tuple[torch.Tensor, torch.Tensor] = precompute_rope_angle(
            args.dim_ // args.n_heads_, args.max_seq_len_, args.device)

        self.norm_eps_ = args.norm_eps_

        self.device_ = args.device
        self.n_heads_ = args.n_heads_
        self.vocab_size_ = args.vocab_size_
        self.pad_token_id_ = args.pad_token_id_
        self.dim_ = args.dim_

        # need to set
        self.eos_token_id_ = -1

    # train model: output is probs
    # inference model: output is tokens
    def forward(self, input: MultiLoraBatchData) -> torch.Tensor:
        tokens = torch.tensor(input.batch_tokens_,
                              dtype=torch.int64).to(self.device_)
        if input.inference_model_:
            mask = None
            input_mask = tokens != self.pad_token_id_
        else:
            mask = precompute_mask(input, self.n_heads_, self.device_)

        # only for inference
        if input.inference_model_:
            start_idx = 0
            need_inference_row = list(range(0, tokens.shape[0]))

            for end_idx in range(input.min_token_size_, input.max_cutoff_len_):
                if len(need_inference_row) <= 0:
                    break
                # [start_idx, end_idx)
                row_index = torch.as_tensor(
                    need_inference_row, dtype=torch.long, device=self.device_)
                col_index = torch.arange(
                    start_idx, end_idx, dtype=torch.long, device=self.device_)

                input_tokens = torch.index_select(tokens, 0, row_index)
                input_tokens = torch.index_select(input_tokens, 1, col_index)

                input_data = F.embedding(input_tokens, self.token_embedding_,
                                         padding_idx=self.pad_token_id_)
                for layer in self.layers_:
                    input_data = layer.forward(
                        input_data, mask, self.rope_angle_, input)

                input_data = self.norm_.forward(input_data)
                # only get the last be the predict token
                input_data = input_data[:, -1,
                                        :] @ self.output_.transpose(0, 1)
                # the predict output is input_data
                next_token = torch.argmax(input_data, dim=-1)
                next_token = torch.where(
                    input_mask[need_inference_row, end_idx],
                    tokens[need_inference_row, end_idx], next_token)
                # attach to result
                tokens[need_inference_row, end_idx] = next_token
                # delete the end sentence
                delete_inference_idx = []
                for idx, item in enumerate(next_token):
                    if item.item() == self.eos_token_id_:
                        delete_inference_idx.append(idx)
                for idx in reversed(delete_inference_idx):
                    need_inference_row.pop(idx)
                    # delete the cache-kv
                    for layer_id, _ in enumerate(self.layers_):
                        input.cache_key_[layer_id] = torch.cat(
                            (input.cache_key_[layer_id][:idx, ...], input.cache_key_[layer_id][idx + 1:, ...]))
                        input.cache_value_[layer_id] = torch.cat(
                            (input.cache_value_[layer_id][:idx, ...], input.cache_value_[layer_id][idx + 1:, ...]))
                # need early stop when the next_token is end
                start_idx = end_idx
            return tokens

        # only for train

        data = F.embedding(tokens, self.token_embedding_,
                           padding_idx=self.pad_token_id_).requires_grad_(True)

        def create_forward_for_checkpoint(module: Transformer):
            def forward_for_checkpoint(*inputs):
                return module.forward(*inputs)
            return forward_for_checkpoint

        for layer in self.layers_:
            # use CheckpointOffloadFunction to use offload mode
            data = CheckpointRecomputeFunction.apply(create_forward_for_checkpoint(
                layer), data, mask, self.rope_angle_, input)

        data = self.norm_.forward(data)
        data @= self.output_.transpose(0, 1)

        return data

    def init_random_lora_weight(self, adapter_name: str,
                                r: int,
                                lora_alpha: int,
                                lora_dropout: float,
                                target: Dict[str, bool]):
        for transformer_layer in self.layers_:
            transformer_layer.init_lora_layer_weight(
                adapter_name, r, lora_alpha, lora_dropout, target)

    def from_pretrained(path: str,
                        device: str,
                        bits: int = None,
                        fp16: bool = True,
                        bf16: bool = True,
                        double_quant: bool = True,
                        quant_type: str = 'nf4',
                        log_fn=None) -> LLMModel:
        if bits in [4, 8]:
            if log_fn is not None:
                log_fn('Loading model with quantization, bits = %i' % bits)
            from transformers import BitsAndBytesConfig
            compute_dtype = (torch.float16 if fp16 else (
                torch.bfloat16 if bf16 else torch.float32))
            llama_model = LlamaForCausalLM.from_pretrained(
                path,
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                device_map=device,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=bits == 4,
                    load_in_8bit=bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=double_quant,
                    bnb_4bit_quant_type=quant_type,
                ),
                torch_dtype=(torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32)))
        else:
            llama_model = LlamaForCausalLM.from_pretrained(
                path,
                device_map=device,
                torch_dtype=torch.float32)

        llama_args = LLMModelArgs()
        llama_args.dim_ = llama_model.config.hidden_size
        llama_args.n_heads_ = llama_model.config.num_attention_heads
        llama_args.n_kv_heads_ = llama_args.n_heads_ if not hasattr(
            llama_model.config, "num_key_value_heads") else llama_model.config.num_key_value_heads
        llama_args.n_layers_ = llama_model.config.num_hidden_layers
        llama_args.norm_eps_ = llama_model.config.rms_norm_eps
        llama_args.vocab_size_ = llama_model.config.vocab_size
        llama_args.max_seq_len_ = 4096 if not hasattr(
            llama_model.config, "max_sequence_length") else llama_model.config.max_sequence_length
        llama_args.pad_token_id_ = -1
        llama_args.device = device

        model = LlamaModel(llama_args)

        model.token_embedding_ = llama_model.model.embed_tokens.weight.to(
            device=device).requires_grad_(False)
        model.output_ = llama_model.lm_head.weight.to(
            dtype=torch.float32, device=device).requires_grad_(False)
        model.norm_ = RMSNorm(llama_model.model.norm.weight.to(
            device=device).requires_grad_(False), model.norm_eps_)

        for idx, layer in enumerate(llama_model.model.layers):
            model.layers_[idx].wq_ = Linear(
                layer.self_attn.q_proj, device=device)
            model.layers_[idx].wk_ = Linear(
                layer.self_attn.k_proj, device=device)
            model.layers_[idx].wv_ = Linear(
                layer.self_attn.v_proj, device=device)
            model.layers_[idx].wo_ = Linear(
                layer.self_attn.o_proj, device=device)
            model.layers_[idx].w1_ = Linear(layer.mlp.gate_proj, device=device)
            model.layers_[idx].w2_ = Linear(layer.mlp.down_proj, device=device)
            model.layers_[idx].w3_ = Linear(layer.mlp.up_proj, device=device)
            model.layers_[idx].attention_norm_ = RMSNorm(
                layer.input_layernorm.weight.to(device=device).requires_grad_(False), model.norm_eps_)
            model.layers_[idx].ffn_norm_ = RMSNorm(
                layer.post_attention_layernorm.weight.to(device=device).requires_grad_(False), model.norm_eps_)

        return model

    def get_train_paramas(self, config: Dict[str, str]) -> Dict[str, List[torch.Tensor]]:
        train_paramas = {}

        for transformer_layer in self.layers_:
            for lora_config in config["lora"]:
                adapter_name = lora_config["name"]
                if adapter_name not in train_paramas:
                    train_paramas[adapter_name] = []

                lora_layer_list = [transformer_layer.wq_.loras_, transformer_layer.wk_.loras_,
                                   transformer_layer.wv_.loras_, transformer_layer.wo_.loras_,
                                   transformer_layer.w1_.loras_, transformer_layer.w2_.loras_,
                                   transformer_layer.w3_.loras_]

                for lora_layer in lora_layer_list:
                    if adapter_name in lora_layer:
                        train_paramas[adapter_name].append(
                            lora_layer[adapter_name].lora_a_)
                        train_paramas[adapter_name].append(
                            lora_layer[adapter_name].lora_b_)

        return train_paramas