import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import TransposeLast, Fp32GroupNorm, Fp32LayerNorm
from fairseq.utils import buffered_arange
from sentence_transformers import SentenceTransformer, util
from gumbel_vector_quantizer import GumbelVectorQuantizer
from kmeans_vector_quantizer import KmeansVectorQuantizer
from transformers import AutoTokenizer, AutoModel


class CPC(nn.Module):
    def __init__(self, args):
        super(CPC, self).__init__()

        # 1D conv Encoder to get the outputs with an over-all stride of 4
        feature_enc_layers = eval(args.conv_feature_layers)
        offset = self.compute_offset(feature_enc_layers)
        # ## replacing conv encoder with pretrained Sentence Transformer .. must freeze it
        self.encoder = AutoModel.from_pretrained(
                'distilbert-base-uncased'
            )
        
        ## freeze text encoder
        for n, p in self.encoder.named_parameters():
            p.requires_grad =False
        


        # Quantizer
        self.vector_quantizer = None
        if args.vq_type == "kmeans":
            self.vector_quantizer = KmeansVectorQuantizer(
                dim=768,
                num_vars=args.num_vars,
                groups=args.groups,
                combine_groups=False,
                vq_dim=768,
                time_first=False,
                gamma=args.vq_gamma,
            )
        elif args.vq_type == "gumbel":
            self.vector_quantizer = GumbelVectorQuantizer(
                dim=768,
                num_vars=args.num_vars,
                temp=(args.gumbel_temperature, 0.5, 0.999988),
                groups=args.groups,
                combine_groups=False,
                vq_dim=768,
                time_first=False,
                activation=nn.ReLU(),
            )

        # Summarizing the features over time
        if args.aggregator_type == "conv":
            agg_layers = eval(args.conv_aggregator_layers)
            agg_layers = agg_layers[: args.num_conv_agg_layers]
            print("final agg layers: {}".format(agg_layers))

            self.aggregator = ConvAggegator(
                conv_layers=agg_layers,
                embed=768,  # from the conv feature encoder
                dropout=0,
                skip_connections=True,
                residual_scale=0.5,
                non_affine_group_norm=False,
                conv_bias=True,
                zero_pad=False,
                activation=nn.ReLU(),
            )
        elif args.aggregator_type == "gru":
            self.aggregator = nn.Sequential(
                TransposeLast(),
                nn.GRU(
                    input_size=768,
                    hidden_size=768,
                    num_layers=2,
                    dropout=0.2,
                ),
                TransposeLast(deconstruct_idx=0),
            )

        # Prediction model
        self.prediction_model = PredictionModel(
            in_dim=768,  # the gru output dim or conv agg. last channel dim
            out_dim=768,  # conv encoder channels
            prediction_steps=args.num_steps_prediction,  # prediction horizon
            n_negatives=args.num_negatives,  # number of negatives
            cross_sample_negatives=0,
            sample_distance=None,
            dropout=0,
            offset=offset,  # how many steps to skip during future prediction?
            balanced_classes=False,
            infonce=True,
        )

    def forward(self, text_inputs, attention_mask):
        result = {}

        ##
        batch_size = text_inputs.shape[0]
      
        text_inputs = text_inputs.reshape(-1, text_inputs.shape[-1])
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
       
        out = self.encoder(text_inputs,attention_mask=attention_mask)[0]
        out = torch.mean(out, dim=1)
        features = out.reshape(batch_size, -1, out.shape[-1]).transpose(1,2)
    
    
        # Vector quantizer (if we are using one)
        if self.vector_quantizer:
            q_res = self.vector_quantizer(features)
            features = q_res["x"]
            for k in q_res.keys():
                if k != "x":
                    result[k] = q_res[k]

        # Predicting the future timesteps
        agg = self.aggregator(features)

        # Making future predictions
        predictions, targets = self.prediction_model(agg, features)

        result["logits"] = predictions
        result["targets"] = targets
        # print('Current gumbel temp: {}'.format(
        # self.vector_quantizer.curr_temp))

        return result

    def compute_offset(self, feature_enc_layers):
        jin = 0
        rin = 0
        for _, k, stride in feature_enc_layers:
            if rin == 0:
                rin = k
            rin = rin + (k - 1) * jin
            if jin == 0:
                jin = stride
            else:
                jin *= stride
        offset = math.ceil(rin / jin)

        offset = int(offset)

        return offset


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.encoder = Convolutional1DEncoder(args)

    def forward(self, inputs):
        return self.encoder(inputs)


class Convolutional1DEncoder(nn.Module):
    def __init__(self, args):
        super(Convolutional1DEncoder, self).__init__()
        if args.input_downsampling == 2:
            self.encoder = nn.Sequential(
                ConvBlock(31, 64, kernel_size=4, stride=2),
                ConvBlock(64, 128, kernel_size=1, stride=1),
                ConvBlock(128, 256, kernel_size=1, stride=1),
                ConvBlock(256, 768, kernel_size=1, stride=1),
            )
        elif args.input_downsampling == 4:
            self.encoder = nn.Sequential(
                ConvBlock(31, 64, kernel_size=4, stride=2),
                ConvBlock(64, 128, kernel_size=4, stride=2),
                ConvBlock(128, 256, kernel_size=1, stride=1),
                ConvBlock(256, 768, kernel_size=1, stride=1),
            )
        elif args.input_downsampling == 1:
            self.encoder = nn.Sequential(
                ConvBlock(31, 64, kernel_size=4, stride=1),
                ConvBlock(64, 128, kernel_size=1, stride=1),
                ConvBlock(128, 256, kernel_size=1, stride=1),
                ConvBlock(256, 768, kernel_size=1, stride=1),
            )

        num_parameters = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        print(
            "The number of trainable parameters in the conv enc is {}".format(
                num_parameters
            )
        )

    def forward(self, inputs):
        # Tranposing since the Conv1D requires
        inputs = inputs.permute(0, 2, 1)
        encoder = self.encoder(inputs)
        # encoder = encoder.permute(0, 2, 1)

        return encoder


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, dropout_prob=0.2
    ):
        super(ConvBlock, self).__init__()

        # 1D convolutional layer
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

        self.dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        conv = self.conv(inputs)
        relu = self.relu(conv)
        dropout = self.dropout(relu)

        return dropout


class ZeroPad1d(nn.Module):
    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        return F.pad(x, (self.pad_left, self.pad_right))


def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)

    return mod


class ConvAggegator(nn.Module):
    def __init__(
        self,
        conv_layers,
        embed,
        dropout,
        skip_connections,
        residual_scale,
        non_affine_group_norm,
        conv_bias,
        zero_pad,
        activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            # padding dims only really make sense for stride = 1
            ka = k // 2
            kb = ka - 1 if k % 2 == 0 else ka

            pad = (
                ZeroPad1d(ka + kb, 0) if zero_pad else nn.ReplicationPad1d((ka + kb, 0))
            )

            return nn.Sequential(
                pad,
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias),
                nn.Dropout(p=dropout),
                norm_block(False, n_out, affine=not non_affine_group_norm),
                activation,
            )

        in_d = embed
        self.conv_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        for dim, k, stride in conv_layers:
            if in_d != dim and skip_connections:
                self.residual_proj.append(nn.Conv1d(in_d, dim, 1, bias=False))
            else:
                self.residual_proj.append(None)

            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if rproj is not None:
                    residual = rproj(residual)
                x = (x + residual) * self.residual_scale
        return x


class PredictionModel(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        prediction_steps,
        n_negatives,
        cross_sample_negatives,
        sample_distance,
        dropout,
        offset,
        balanced_classes,
        infonce,
    ):
        super(PredictionModel, self).__init__()
        self.n_negatives = n_negatives
        self.cross_sample_negatives = cross_sample_negatives
        self.sample_distance = sample_distance
        self.project_to_steps = nn.ConvTranspose2d(
            in_dim, out_dim, (1, prediction_steps)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.offset = offset
        self.balanced_classes = balanced_classes
        self.infonce = infonce

    def sample_negatives(self, y):
        bsz, fsz, tsz = y.shape

        y = y.transpose(0, 1)  # BCT -> CBT
        y = y.contiguous().view(fsz, -1)  # CBT => C(BxT)

        cross_high = tsz * bsz
        high = tsz if self.sample_distance is None else min(tsz, self.sample_distance)
        assert high > 1

        neg_idxs = torch.randint(low=0, high=high, size=(bsz, self.n_negatives * tsz))

        with torch.no_grad():
            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(tsz)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * tsz)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(tsz)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * tsz),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[..., neg_idxs.view(-1)]
        negs = negs.view(
            fsz, bsz, self.n_negatives + self.cross_sample_negatives, tsz
        ).permute(
            2, 1, 0, 3
        )  # to NxBxCxT

        return negs

    def forward(self, x, y):
        x = x.unsqueeze(-1)
        x = self.project_to_steps(x)  # BxCxTxS
        x = self.dropout(x)

        negatives = self.sample_negatives(y)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)  # Copies x B x C x T

        copies = targets.size(0)
        bsz, dim, tsz, steps = x.shape
        steps = min(steps, tsz - self.offset)

        predictions = x.new(
            bsz * copies * (tsz - self.offset + 1) * steps
            - ((steps + 1) * steps // 2) * copies * bsz
        )
        if self.infonce:
            labels = predictions.new_full(
                (predictions.shape[0] // copies,), 0, dtype=torch.long
            )
        else:
            labels = torch.zeros_like(predictions)
        weights = (
            torch.full_like(labels, 1 / self.n_negatives)
            if self.balanced_classes and not self.infonce
            else None
        )

        start = end = 0
        for i in range(steps):
            offset = i + self.offset
            end = start + (tsz - offset) * bsz * copies
            if self.infonce:
                predictions[start:end] = torch.einsum(
                    "bct,nbct->tbn", x[..., :-offset, i], targets[..., offset:]
                ).flatten()
            else:
                pos_num = (end - start) // copies
                predictions[start:end] = torch.einsum(
                    "bct,nbct->nbt", x[..., :-offset, i], targets[..., offset:]
                ).flatten()
                labels[start : start + pos_num] = 1.0
                if weights is not None:
                    weights[start : start + pos_num] = 1.0
            start = end
        assert end == predictions.numel(), "{} != {}".format(end, predictions.numel())

        if self.infonce:
            predictions = predictions.view(-1, copies)
        else:
            if weights is not None:
                labels = (labels, weights)

        return predictions, labels


class QuantizedFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(QuantizedFeatureExtractor, self).__init__()
        # Encoder
        self.encoder = AutoModel.from_pretrained(
                'distilbert-base-uncased'
            )
        
        ## freeze text encoder
        for n, p in self.encoder.named_parameters():
            p.requires_grad =False
        # Quantization modules
        self.vector_quantizer = None
        if args.vq_type == "kmeans":
            self.vector_quantizer = KmeansVectorQuantizer(
                dim=768,
                num_vars=args.num_vars,
                groups=args.groups,
                combine_groups=False,
                vq_dim=768,
                time_first=False,
                gamma=args.vq_gamma,
            )
        elif args.vq_type == "gumbel":
            self.vector_quantizer = GumbelVectorQuantizer(
                dim=768,
                num_vars=args.num_vars,
                temp=(2.0, 0.5, 0.999988),
                groups=args.groups,
                combine_groups=False,
                vq_dim=768,
                time_first=False,
                activation=nn.ReLU(),
            )

    def forward(self, text_inputs, attention_mask):
        result = {}

        # Encoding through the conv layers
        batch_size = text_inputs.shape[0]
      
        text_inputs = text_inputs.reshape(-1, text_inputs.shape[-1])
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
       
        out = self.encoder(text_inputs,attention_mask=attention_mask)[0]
        out = torch.mean(out, dim=1)
        features = out.reshape(batch_size, -1, out.shape[-1]).transpose(1,2)

        # Vector quantizer
        q_res = self.vector_quantizer(features, produce_targets=True)

        for k in q_res.keys():
            if k != "x":
                result[k] = q_res[k]

        return result

    def load_pretrained_weights(self, args):
        state_dict_path = os.path.join(args.saved_model)

        print("Loading the pre-trained weights")
        checkpoint = torch.load(state_dict_path, map_location=args.device)
        pretrained_checkpoint = checkpoint["model_state_dict"]

        model_dict = self.state_dict()

        # What weights are *not* copied
        missing = {
            k: v for k, v in pretrained_checkpoint.items() if k not in model_dict
        }
        print(
            "The weights from saved model not in classifier are: {}".format(
                missing.keys()
            )
        )

        missing = {
            k: v for k, v in model_dict.items() if k not in pretrained_checkpoint
        }
        print(
            "The weights from classifier not in the saved model are: {}".format(
                missing.keys()
            )
        )

        self.load_state_dict(pretrained_checkpoint, False)

        return


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(
                'distilbert-base-uncased'
            )
        
        ## freeze text encoder
        for n, p in self.encoder.named_parameters():
            p.requires_grad =False

        # Convolutional aggregator to summarize the features
        self.aggregator_type = args.aggregator_type
        if args.aggregator_type == "conv":
            agg_layers = eval(args.conv_aggregator_layers)
            agg_layers = agg_layers[: args.num_conv_agg_layers]
            agg_dim = agg_layers[-1][0]
            print("final agg layers: {}".format(agg_layers))

            self.aggregator = ConvAggegator(
                conv_layers=agg_layers,
                embed=768,  # from the conv feature encoder
                dropout=0,
                skip_connections=True,
                residual_scale=0.5,
                non_affine_group_norm=False,
                conv_bias=True,
                zero_pad=False,
                activation=nn.ReLU(),
            )
        else:
            agg_dim = 768
            self.aggregator = nn.Sequential(
                TransposeLast(),
                nn.GRU(
                    input_size=768,
                    hidden_size=agg_dim,
                    num_layers=2,
                    dropout=0.2,
                ),
            )

        # Softmax
        self.softmax = nn.Sequential(nn.Linear(agg_dim, 256),
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(256, 128),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(128, args.num_classes))

    def forward(self, text_inputs, attention_mask):
        batch_size = text_inputs.shape[0]
      
        text_inputs = text_inputs.reshape(-1, text_inputs.shape[-1])
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
       
        out = self.encoder(text_inputs,attention_mask=attention_mask)[0]
        out = torch.mean(out, dim=1)
        encoder = out.reshape(batch_size, -1, out.shape[-1]).transpose(1,2)

        # Global max pooling for classification
        if self.aggregator_type == "conv":
            aggregator = self.aggregator(encoder)
            pool = F.max_pool1d(aggregator, kernel_size=aggregator.shape[2])
            pool = pool.squeeze(2)
        else:
            aggregator, _ = self.aggregator(encoder)
            pool = aggregator[:, -1, :]

        softmax = self.softmax(pool)

        return softmax

    def load_pretrained_weights(self, args):
        state_dict_path = os.path.join(args.saved_model)

        print("Loading the pre-trained weights")
        checkpoint = torch.load(state_dict_path, map_location=args.device)
        pretrained_checkpoint = checkpoint["model_state_dict"]

        model_dict = self.state_dict()

        # What weights are *not* copied
        missing = {
            k: v for k, v in pretrained_checkpoint.items() if k not in model_dict
        }
        print(
            "The weights from saved model not in classifier are: {}".format(
                missing.keys()
            )
        )

        missing = {
            k: v for k, v in model_dict.items() if k not in pretrained_checkpoint
        }
        print(
            "The weights from classifier not in the saved model are: {}".format(
                missing.keys()
            )
        )

        self.load_state_dict(pretrained_checkpoint, False)

        return

    def freeze_encoder_layers(self):
        """
        To set only the softmax to be trainable
        :return: None, just setting the encoder part (or the CPC model) as
        frozen
        """
        print("Setting only the softmax layer to be trainable")
        num_parameters = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        print(
            "Before setting, the number of trainable parameters in the "
            "encoder is {}".format(num_parameters)
        )

        # First setting the feature extractor and aggregator's to eval
        self.encoder.eval()
        self.aggregator.eval()

        # Then setting the requires_grad to False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.aggregator.parameters():
            param.requires_grad = False

        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            "After setting, the number of trainable parameters in the "
            "model is {}".format(num_parameters)
        )

        return


class QuantizedClassifier(nn.Module):
    def __init__(self, args):
        super(QuantizedClassifier, self).__init__()
        # Randomly initialized embeddings
        self.embeddings = nn.Embedding(args.vocab_size, args.embedding_size)
        self.embeddings.weight.requires_grad = True

        # RNN model
        if args.classifier_type == 'gru':
            self.rnn_1 = nn.GRU(input_size=args.embedding_size,
                                hidden_size=128,
                                num_layers=2,
                                bidirectional=False,
                                batch_first=True,
                                dropout=0.2)

        elif args.classifier_type == 'lstm':
            self.rnn_1 = nn.LSTM(input_size=args.embedding_size,
                                 hidden_size=128,
                                 num_layers=2,
                                 bidirectional=False,
                                 batch_first=True,
                                 dropout=0.2)

        shape = 128
        self.classifier = nn.Sequential(nn.Linear(shape, 256),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(256, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(128, args.num_classes))

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)

        r_out, _ = self.rnn_1(embeddings, None)

        softmax = self.classifier(r_out[:, -1, :])

        return softmax
    
    def load_pretrained_weights(self, args):    
        state_dict_path = os.path.join(args.saved_model)

        print("Loading the pre-trained weights", state_dict_path)
        checkpoint = torch.load(state_dict_path, map_location=args.device)
        pretrained_checkpoint = checkpoint["model_state_dict"]

        model_dict = self.state_dict()

        # What weights are *not* copied
        missing = {
            k: v for k, v in pretrained_checkpoint.items() if k not in model_dict
        }
        print(
            "The weights from saved model not in classifier are: {}".format(
                missing.keys()
            )
        )

        missing = {
            k: v for k, v in model_dict.items() if k not in pretrained_checkpoint
        }
        print(
            "The weights from classifier not in the saved model are: {}".format(
                missing.keys()
            )
        )

        self.load_state_dict(pretrained_checkpoint, False)

        return
    
    