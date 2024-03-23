import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel, BertConfig, BertLayer, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder, BertLayer, BertOutput, BertAttention, BertSelfAttention, BertSelfOutput, BertIntermediate, BertPreTrainedModel, BertEmbeddings, BertPooler 
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch
from typing import Optional, Tuple, Union, List
import math
import torch.nn.functional as F
from torch.autograd import Variable

class ModelIAS(nn.Module):

    def __init__(
            self,
            out_slot,
            out_int,
            device,
            freeze=False,
            dropout=0.0
    ):
        """
        Modified ModelIAS
        :param hid_size: Hidden size
        :param out_slot: number of slots (output size for slot filling)
        :param out_int: number of intents (ouput size for intent class)
        :param emb_size: word embedding size
        :param vocab_len: vocabulary size
        :param n_layer: number of layers for the LSTM
        :param pad_index: padding index
        """

        super(ModelIAS, self).__init__()

        # Bert has its own embeddings
        # self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        # Bert tokenizer, do not finetune it for the moment
        self.bert = HebbBertModel.from_pretrained("bert-base-uncased").to(device)#requires_grad_(not freeze).to(device)

        # hidden size for bert is 768
        self.hid_size = 768
        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(self.hid_size, out_slot)
        self.intent_out = nn.Linear(self.hid_size, out_int)

    def forward(self, utterance, seq_lengths):

        # create the attention mask, in order to avoid use the padding
        attention_mask = torch.zeros(size=(len(seq_lengths), max(seq_lengths))).to(utterance.device)
        for i, seq_length in enumerate(seq_lengths):
            attention_mask[i, :seq_length] = 1

        # UTTERANCE: [BATCH_SIZE, SEQ_LEN] -> [128, 32]
        bert_out = self.bert(utterance)  # attention_mask=attention_mask)

        # This is the CLS token to use for intent classification
        cls = bert_out.pooler_output  # [128, 768] -> [batch size, hid size]
        # This is the last hidden state to use for slot filling
        utt_encoded = bert_out.last_hidden_state  # [128, 32, 768] -> [batch size, seq len, hid size]

        # Get intent classification
        intent = self.intent_out(self.dropout(cls))

        # Slot filling
        slots = self.slot_out(self.dropout(utt_encoded))

        # Slot size: seq_len, batch size, calsses
        slots = slots.permute(0, 2, 1)  # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
    
    def reset_hebb(self):
        # reset all the parameters containing the trace
        for name, param in self.bert.named_parameters():
            if 'trace' in name:
                nn.init.zeros_(param)    
    
class HebbBase(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.trace = nn.Parameter(torch.zeros((in_features, out_features), requires_grad=False))
        self.Ci = nn.Parameter(torch.ones(in_features) * .1)
        self.Cj = nn.Parameter(torch.ones(out_features) * .1)
        self.eta = 0.1
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.trace.requires_grad_(False)

    def reset_trace(self):
        nn.init.zeros_(self.trace.weight)

    def hebb_update(self, input, output):
        CiCj = torch.einsum('i,j->ij', self.Ci, self.Cj)
        if len(input.shape) == 3:
            prepost = torch.einsum('bli,blo->io', input, output)
        else:
            prepost = torch.einsum('bi,bo->io', input, output)
        CiCj = CiCj
        hebb = (prepost * CiCj).mean(dim=0)

        self.trace.data = (1 - self.eta) * self.trace + self.eta * hebb
        self.trace.data = self.trace.data / self.trace.data.norm()
        return self.alpha * (input @ self.trace)
    
    def reset_hebb_trace(self):
        nn.init.zeros_(self.trace)


class HebbBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([HebbBertLayer(config) for _ in range(config.num_hidden_layers)])

class HebbBertLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = HebbBertAttention(config)
        self.output = HebbBertOutput(config)
        self.intermediate = HebbBertIntermediate(config)

class HebbBertOutput(BertOutput):
    def __init__(self, config):
        super().__init__(config)
        self.hebb = HebbBase(config.intermediate_size, config.hidden_size)
        self.dense.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.dense(hidden_states)
        hebb_res = self.hebb.hebb_update(hidden_states, output)
        output = self.dropout(output + hebb_res)
        output = self.LayerNorm(output + input_tensor)
        return output  

class HebbBertAttention(BertAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config)
        self.self = HebbBertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = HebbBertSelfOutput(config)

class HebbBertSelfOutput(BertSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        self.hebb = HebbBase(config.hidden_size, config.hidden_size)
        self.dense.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.dense(hidden_states)
        hebb_res = self.hebb.hebb_update(input_tensor, output)
        output = self.dropout(output + hebb_res)
        output = self.LayerNorm(output + input_tensor)
        return output
    
class HebbBertSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.hebb_key = HebbBase(config.hidden_size, self.all_head_size)
        self.hebb_value = HebbBase(config.hidden_size, self.all_head_size)
        self.hebb_query = HebbBase(config.hidden_size, self.all_head_size)

        self.key.requires_grad_(False)
        self.value.requires_grad_(False)
        self.query.requires_grad_(False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states) 
        mixed_query_layer = mixed_query_layer + self.hebb_query.hebb_update(hidden_states, mixed_query_layer)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.key(encoder_hidden_states)
            key_layer = key_layer + self.hebb_key.hebb_update(encoder_hidden_states, key_layer)
            key_layer = self.transpose_for_scores(key_layer)

            value_layer = self.value(encoder_hidden_states)
            value_layer = value_layer + self.hebb_value.hebb_update(encoder_hidden_states, value_layer)
            value_layer = self.transpose_for_scores(value_layer)

            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.key(hidden_states)
            key_layer = key_layer + self.hebb_key.hebb_update(hidden_states, key_layer)
            key_layer = self.transpose_for_scores(key_layer)

            value_layer = self.value(hidden_states)
            value_layer = value_layer + self.hebb_value.hebb_update(hidden_states, value_layer)
            value_layer = self.transpose_for_scores(value_layer)

            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.key(hidden_states)
            key_layer = key_layer + self.hebb_key.hebb_update(hidden_states, key_layer)
            key_layer = self.transpose_for_scores(key_layer)

            value_layer = self.value(hidden_states)
            value_layer = value_layer + self.hebb_value.hebb_update(hidden_states, value_layer)
            value_layer = self.transpose_for_scores(value_layer)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    
class HebbBertIntermediate(BertIntermediate):
    def __init__(self, config):
        super().__init__(config)
        self.hebb = HebbBase(config.hidden_size, config.intermediate_size)
        self.dense.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.dense(hidden_states)
        hebb_res = self.hebb.hebb_update(hidden_states, out)
        hidden_states = self.intermediate_act_fn(out + hebb_res)
        return hidden_states
        
class HebbBertPooler(BertPooler):
    def __init__(self, config):
        BertPooler.__init__(self, config)
        self.hebb = HebbBase(config.hidden_size, config.hidden_size)
        self.dense.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        hebb_res = self.hebb.hebb_update(first_token_tensor, pooled_output)
        pooled_output = self.activation(pooled_output + hebb_res)
        return pooled_output
    
class HebbBertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = HebbBertEncoder(config)
        self.embeddings.requires_grad_(False)

        self.pooler = HebbBertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
