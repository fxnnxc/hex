import torch
from functools import partial

from .model import *

def bert_attention_intervention(model, inputs, attention_weights, target_position_mask=None, target_head_list=None):
	def intervention_hook(module, inputs, outputs, attn_override, attn_override_mask):
		attention_override_module = BertSelfAttentionOverride(module, attn_override, attn_override_mask)
		override_outputs = attention_override_module(*inputs)
		outputs[0][:] = override_outputs[0]
		outputs[1][:] = override_outputs[1]


	with torch.no_grad():
		hooks = []
		for layer_num in range(model.config.num_hidden_layers):
			attn_override = attention_weights[layer_num]
			attn_override_mask = torch.zeros_like(attn_override, dtype=torch.uint8)

			if target_head_list is None:
				attn_override_mask += 1
			elif layer_num in target_head_list:
				for head_idx in target_head_list[layer_num]:
					attn_override_mask[:, head_idx, :, :] = 1

			if target_position_mask is not None:
				attn_override_mask *= target_position_mask

			hooks.append(model.bert.encoder.layer[layer_num].attention.self.register_forward_hook(partial(intervention_hook, attn_override=attn_override, attn_override_mask=attn_override_mask)))
		
		outputs = model(**inputs, output_attentions=True)

		for hook in hooks:
			hook.remove()

	return outputs


def roberta_attention_intervention(model, inputs, attention_weights, target_position_mask=None, target_head_list=None):
	def intervention_hook(module, inputs, outputs, attn_override, attn_override_mask):
		attention_override_module = BertSelfAttentionOverride(module, attn_override, attn_override_mask)
		override_outputs = attention_override_module(*inputs)
		outputs[0][:] = override_outputs[0]
		outputs[1][:] = override_outputs[1]


	with torch.no_grad():
		hooks = []
		for layer_num in range(model.config.num_hidden_layers):
			attn_override = attention_weights[layer_num]
			attn_override_mask = torch.zeros_like(attn_override, dtype=torch.uint8)

			if target_head_list is None:
				attn_override_mask += 1
			elif layer_num in target_head_list:
				for head_idx in target_head_list[layer_num]:
					attn_override_mask[:, head_idx, :, :] = 1

			if target_position_mask is not None:
				attn_override_mask *= target_position_mask

			hooks.append(model.roberta.encoder.layer[layer_num].attention.self.register_forward_hook(partial(intervention_hook, attn_override=attn_override, attn_override_mask=attn_override_mask)))
		
		outputs = model(**inputs, output_attentions=True)

		for hook in hooks:
			hook.remove()

	return outputs