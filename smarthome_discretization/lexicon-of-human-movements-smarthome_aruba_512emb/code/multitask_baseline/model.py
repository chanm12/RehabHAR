import os
import torch
import torch.nn.functional as F
from torch import nn

from gumbel_vector_quantizer import GumbelVectorQuantizer
from kmeans_vector_quantizer import KmeansVectorQuantizer


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        # Weights initialization
        def _weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.GRU, nn.LSTM)):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, x):
        x = self.dropout(self.relu(self.conv(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Conv encoders
        self.conv1 = ConvBlock(
            in_channels=args.input_size, out_channels=32, kernel_size=24
        )
        self.conv2 = ConvBlock(in_channels=32, out_channels=64, kernel_size=16)
        self.conv3 = ConvBlock(in_channels=64, out_channels=96, kernel_size=8)

        # Quantization module
        self.quant_emb = nn.Linear(96, 128)
        if args.quantization_method == "kmeans":
            self.quantizer = KmeansVectorQuantizer(
                dim=128,
                num_vars=args.num_vars,
                groups=args.groups,
                combine_groups=False,
                vq_dim=128,
                time_first=True,
                gamma=0.25,
            ).to(args.device)
        elif args.quantization_method == "gumbel":
            self.quantizer = GumbelVectorQuantizer(
                dim=128,
                num_vars=args.num_vars,
                temp=(2.0, 0.5, 0.999995),
                groups=args.groups,
                combine_groups=False,
                vq_dim=128,
                time_first=True,
            ).to(args.device)
        self.quant_de_emb = nn.Linear(128, 96)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv encoding
        x = x.squeeze(1).transpose(1, 2)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        # print('conv: {}'.format(conv3.shape))
        x = conv3.transpose(1, 2)
        # print('transpose: {}'.format(x.shape))

        # Quantization
        quant_emb = self.relu(self.quant_emb(x))
        # print('quant emb: {}'.format(quant_emb.shape))
        quantizer_all = self.quantizer(quant_emb, produce_targets=True)
        quantizer = quantizer_all["x"]
        # print('quantizer: {}'.format(quant_emb.shape))
        quant_de_emb = self.relu(self.quant_de_emb(quantizer)).transpose(1, 2)
        # print('quant de emb: {}'.format(quant_de_emb.shape))

        # Global Max Pooling (as per
        # https://github.com/keras-team/keras/blob
        # /7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/layers/pooling.py
        # #L559) for 'channels_first'
        x = F.max_pool1d(quant_de_emb, kernel_size=quant_de_emb.shape[2])
        x = x.squeeze(2)

        return x, quantizer_all

    def load_pretrained_weights(self, args):
        state_dict_path = os.path.join(args.saved_model)

        print("Loading the pre-trained weights")
        pretrained_checkpoint = torch.load(state_dict_path, map_location=args.device)

        updated_checkpoints = {}
        for k, v in pretrained_checkpoint.items():
            updated_checkpoints[k] = v

        model_dict = self.state_dict()

        # What weights are *not* copied
        missing = {k: v for k, v in updated_checkpoints.items() if k not in model_dict}
        print("The mismatched weights are: {}".format(missing.keys()))

        self.load_state_dict(updated_checkpoints, False)

        return


class TaskClassifier(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(out_channels, 2)

    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.classifier(x)
        if not self.training:
            x = torch.sigmoid(x)
        return x


class TPNModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Encoder
        self.encoder = Encoder(args)
        shape = 96

        # Task specific classifiers
        self.task_classifiers = nn.ModuleDict(
            {
                "noised": TaskClassifier(shape),
                "scaled": TaskClassifier(shape),
                "rotated": TaskClassifier(shape),
                "negated": TaskClassifier(shape),
                "horizontally-flipped": TaskClassifier(shape),
                "permuted": TaskClassifier(shape),
                "time-warped": TaskClassifier(shape),
                "channel-shuffled": TaskClassifier(shape),
            }
        )

    def forward(self, x, aug=None):
        # Encoding
        x, quantizer_all = self.encoder(x)

        # Task specific classifier
        output = self.task_classifiers[aug](x)
        return output, quantizer_all


class ClassificationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = Encoder(args=args)
        shape = 96

        # self.classifier = nn.Sequential(
        #     nn.Linear(96, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, args.num_classes)
        # )

        self.classifier = nn.Sequential(
            nn.Linear(shape, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, args.num_classes),
        )

        # # Weights initialization
        def _weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.GRU, nn.LSTM)):
                print(m)
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def load_pretrained_weights(self, args):
        state_dict_path = os.path.join(args.saved_model)

        print("Loading the pre-trained weights")
        pretrained_checkpoint = torch.load(state_dict_path, map_location=args.device)

        updated_checkpoints = {}
        for k, v in pretrained_checkpoint.items():
            updated_checkpoints[k] = v

        model_dict = self.state_dict()

        # What weights are *not* copied
        missing = {k: v for k, v in updated_checkpoints.items() if k not in model_dict}
        print("The mismatched weights are: {}".format(missing.keys()))

        self.load_state_dict(updated_checkpoints, False)

        return

    def freeze_encoder_layers(self):
        """
        To set only the classifier to be trainable
        :return: None, just setting the encoder part as
        frozen
        """
        print("Setting only the softmax layer to be trainable")
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            "Before setting, the number of trainable parameters is {}".format(
                num_parameters
            )
        )

        # First setting the model to eval
        self.encoder.eval()

        # Then setting the requires_grad to False
        for param in self.parameters():
            param.requires_grad = False

        # Setting the classifier layer to training mode
        self.classifier.train()

        # Setting the parameters in the softmax layer to tbe trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            "After setting, the number of trainable parameters is {}".format(
                num_parameters
            )
        )

        return

    def forward(self, x):
        x = self.encoder(x)
        output = self.classifier(x)
        return output


if __name__ == "__main__":
    pass
