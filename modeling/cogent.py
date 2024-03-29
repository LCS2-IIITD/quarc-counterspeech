from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    BartPretrainedModel,
    BartEncoder,
    BartDecoder,
    BartLearnedPositionalEmbedding,
    BartEncoderLayer,
    shift_tokens_right,
    _make_causal_mask,
    _expand_mask 
)
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
    Seq2SeqModelOutput
)
from copy import deepcopy
import torch
from torch import nn
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss, MSELoss, KLDivLoss
import wandb, os
from torch.nn import TransformerEncoder, TransformerEncoderLayer


from .quantizer import VectorQuantizer
from .classification_head import BartClassificationHead
from .perfume import PerFuMe
# from .standard_fusion import SimpleFusion
from .hf_utils import BaseModelOutput
from .positional_encoding import PositionalEncoding



class COGENTEncoder(BartPretrainedModel):
    def __init__(self, config, encoder, codebook, config_params, fusion=None):
        super().__init__(config)
        
        self.sem_encoder = encoder
        self.target_classifier = BartClassificationHead(config.d_model, config.d_model, num_classes=8,
                                pooler_dropout=config.classifier_dropout, reverse_grad=False)
        
        self.pe = PositionalEncoding(config.d_model, max_len=256)
        
        encoder_layers = TransformerEncoderLayer(config.d_model, nhead=12, dim_feedforward=2048, batch_first=True)
        self.contextual_mapper = TransformerEncoder(encoder_layers, num_layers=8)

        self.codebook = nn.Parameter(codebook)
        self.codebook.requires_grad = True

        self.norm = nn.LayerNorm(config.d_model)
        
        self.config_params = config_params
        
        self.target_classification_weight = config_params.target_classification_weight

        self.use_perfume = config_params.use_perfume
        if self.use_perfume:
            if fusion is None:
                k = config_params.k
                dropout = config_params.fusion_dropout
                self.fusion = PerFuMe(config.d_model, config.d_model, k=k, dropout=dropout, config=config_params)
            else:
                self.fusion=fusion
        else:
            self.fusion = None

        self._init_weights(self.target_classifier.dense)
        self._init_weights(self.target_classifier.out_proj)
        
        self.VERBOSE = config_params.verbose
        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        categories = None,
        targets = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        if self.VERBOSE: print(output_hidden_states, output_attentions)
        
        sem_encoder_outputs = self.sem_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        loss = None

        eos_mask = input_ids.eq(self.config.eos_token_id)
        sem_hidden_states = sem_encoder_outputs[0]
        if self.VERBOSE: print(f'sem hidden states shape = {sem_hidden_states.shape}')
        pooled_sem_encoder_outputs = sem_hidden_states[eos_mask, :].view(sem_hidden_states.size(0), -1, sem_hidden_states.size(-1))[:, -1, :]
        if self.VERBOSE: print(f"Pooled Encoder Outputs, shape={pooled_sem_encoder_outputs.shape}")


        bs = input_ids.shape[0]
        e = torch.zeros(bs, 5).to(input_ids.device)
        e.scatter_(1, categories.unsqueeze(1), 1)
        quantized = e@self.codebook
        if self.VERBOSE: print('completed quantization')
        
        
        sem_logits = self.target_classifier(pooled_sem_encoder_outputs)
        if self.VERBOSE: print(f'passed through category classifier, logits shape={sem_logits.shape}')
        loss_fct = CrossEntropyLoss()
        sem_loss = self.target_classification_weight*loss_fct(sem_logits, targets)
        wandb.log({'sem_los': sem_loss})
        loss = sem_loss
        
        
        ### Transformation
        pe_hidden = self.pe(sem_hidden_states)
        contextual_map = self.contextual_mapper(pe_hidden)
        
        contextual_map = sem_hidden_states + contextual_map
        contextual_map = self.norm(contextual_map)
               
    
        hidden_states = self.fusion(contextual_map, quantized)
        if self.VERBOSE: print(f'Completed fusion, shape={hidden_states.shape}')


        encoder_states, all_attentions = None, None
        if len(sem_encoder_outputs) > 1:
            encoder_states = sem_encoder_outputs[1]
        if len(sem_encoder_outputs) > 2:
            all_attentions = sem_encoder_outputs[2]
        
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions, loss] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions, 
            loss=loss
        )
        

class COGENTModel(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = BartEncoder(config, self.shared) # Modified
        self.decoder = BartDecoder(config, self.shared)

        self.config = config
        
        self.VERBOSE = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def update(self, config_params):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clime_components_path = os.path.join(config_params.clime_model_path, 'components')
        
        self.VERBOSE = config_params.verbose
        
        if config_params.use_clime:
            codebook = torch.load(os.path.join(clime_components_path, 'codebook.pt'))
            print("[INFO] Utilizing the codebook from CLIME!")
        elif config_params.use_mean_pooled:
            codebook = torch.load(config_params.pooled_codebook_path)
            print("[INFO] Utilizing the mean-pooled codebook!")
        else:
            codebook = torch.randn(5, 768)
            print("[INFO] Utilizing the randomly initialized codebook!")
            
        codebook = codebook.to(device)
        
        encoder = self.encoder
        decoder = self.decoder
        fusion = None
        
        if config_params.prev_enc_dec:
            print('[INFO] Utilizing the encoder-decoder from CLIME!')
            encoder = torch.load(os.path.join(clime_components_path, 'encoder.pt')).to(device)
            decoder = torch.load(os.path.join(clime_components_path, 'decoder.pt')).to(device)
            
            # for param in encoder.parameters():
            #     param.requires_grad = True
            # for param in decoder.parameters():
            #     param.requires_grad = True
            
        if config_params.freeze_enc_dec:
            assert config_params.prev_enc_dec
            print('[INFO] Freezing the encoder-decoder from CLIME!')
            
            for param in encoder.parameters():
                param.requires_grad = False
            for param in decoder.parameters():
                param.requires_grad = False
        
        if config_params.prev_perfume:
            fusion = torch.load(os.path.join(clime_components_path, 'fusion.pt')).to(device)
            for param in fusion.parameters():
                param.requires_grad = True
        
        modified_encoder = COGENTEncoder(self.config, encoder=encoder, codebook=codebook,
                                         config_params=config_params, fusion=fusion)
        self.encoder = modified_encoder
        self.decoder = decoder

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        categories = None,
        targets = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.VERBOSE: 
            print(f"In forward of base model, categories={categories}")

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                categories = categories,
                targets = targets,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                loss=encoder_outputs[3] if len(encoder_outputs) > 3 else None
            )

        loss = encoder_outputs['loss']
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.VERBOSE: print(f'passed through decoder, shape={decoder_outputs[0].shape}')

        if not return_dict:
            return decoder_outputs + encoder_outputs + (loss, )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        ), loss
        
        
class COGENT(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = COGENTModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()
    
    def update(self, config_params):
        self.model.update(config_params=config_params)

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        categories = None,
        targets = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                print("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs, loss = self.model(
            input_ids,
            attention_mask=attention_mask,
            categories = categories,
            targets = targets,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            wandb.log({'generation_loss': masked_lm_loss.item()})
            masked_lm_loss = masked_lm_loss + loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
