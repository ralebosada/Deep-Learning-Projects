import torch
import torchaudio, torchvision
import os
import matplotlib.pyplot as plt 
import librosa
import argparse
import numpy as np
import wandb
from pytorch_lightning import LightningModule, Trainer, LightningDataModule, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn, einsum
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.datasets.speechcommands import load_speechcommands_item
from einops import rearrange

class SilenceDataset(SPEECHCOMMANDS):
    def __init__(self, root):
        super(SilenceDataset, self).__init__(root, subset='training')
        self.len = len(self._walker) // 35
        path = os.path.join(self._path, torchaudio.datasets.speechcommands.EXCEPT_FOLDER)
        self.paths = [os.path.join(path, p) for p in os.listdir(path) if p.endswith('.wav')]

    def __getitem__(self, index):
        index = np.random.randint(0, len(self.paths))
        filepath = self.paths[index]
        waveform, sample_rate = torchaudio.load(filepath)
        return waveform, sample_rate, "silence", 0, 0

    def __len__(self):
        return self.len

class UnknownDataset(SPEECHCOMMANDS):
    def __init__(self, root):
        super(UnknownDataset, self).__init__(root, subset='training')
        self.len = len(self._walker) // 35

    def __getitem__(self, index):
        index = np.random.randint(0, len(self._walker))
        fileid = self._walker[index]
        waveform, sample_rate, _, speaker_id, utterance_number = load_speechcommands_item(fileid, self._path)
        return waveform, sample_rate, "unknown", speaker_id, utterance_number

    def __len__(self):
        return self.len

class KWSDataModule(LightningDataModule):
    def __init__(self, path, batch_size=2, num_workers=0, n_fft=512, 
                 n_mels=128, win_length=None, hop_length=256, class_dict={},
                 patch_num = 16, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.class_dict = class_dict
        self.patch_num = patch_num

    def prepare_data(self):
        self.train_dataset = torchaudio.datasets.SPEECHCOMMANDS(self.path,
                                                                download=True,
                                                                subset='training')

        silence_dataset = SilenceDataset(self.path)
        unknown_dataset = UnknownDataset(self.path)
        self.train_dataset = torch.utils.data.ConcatDataset([self.train_dataset, silence_dataset, unknown_dataset])
                                                                
        self.val_dataset = torchaudio.datasets.SPEECHCOMMANDS(self.path,
                                                              download=True,
                                                              subset='validation')
        self.test_dataset = torchaudio.datasets.SPEECHCOMMANDS(self.path,
                                                               download=True,
                                                               subset='testing')                                                    
        _, sample_rate, _, _, _ = self.train_dataset[0]
        self.sample_rate = sample_rate
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                              n_fft=self.n_fft,
                                                              win_length=self.win_length,
                                                              hop_length=self.hop_length,
                                                              n_mels=self.n_mels,
                                                              power=2.0)

    def setup(self, stage=None):
        self.prepare_data()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        mels = []
        labels = []
        wavs = []
        for sample in batch:
            waveform, sample_rate, label, speaker_id, utterance_number = sample
            # ensure that all waveforms are 1sec in length; if not pad with zeros
            if waveform.shape[-1] < sample_rate:
                waveform = torch.cat([waveform, torch.zeros((1, sample_rate - waveform.shape[-1]))], dim=-1)
            elif waveform.shape[-1] > sample_rate:
                waveform = waveform[:,:sample_rate]

            # mel from power to db
            temp = ToTensor()(librosa.power_to_db(self.transform(waveform).squeeze().numpy(), ref=np.max))
            mels.append(temp)
            labels.append(torch.tensor(self.class_dict[label]))
            wavs.append(waveform)
        mels = torch.stack(mels)
        labels = torch.stack(labels)
        wavs = torch.stack(wavs)
        mels = rearrange(mels, 'b c (p1 h) (p2 w) -> b (p1 p2) (c h w)', p1=self.patch_num, p2=int(self.patch_num/2))
        return mels, labels

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
      
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias) 
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer) 

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, dim, num_heads, num_blocks, mlp_ratio=4., qkv_bias=False,  
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_ratio, qkv_bias, 
                                     act_layer, norm_layer) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

def init_weights_vit_timm(module: nn.Module):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

class KWSModule(LightningModule):
  def __init__(self, num_classes=37, lr=0.001, max_epochs=5, depth=12, embed_dim=64,
               head=4, patch_dim=192, seqlen=16, **kwargs):
    super().__init__()
    self.save_hyperparameters()
    self.encoder = Transformer(dim=embed_dim, num_heads=head, num_blocks=depth, mlp_ratio=4.,
                               qkv_bias='False', act_layer=nn.GELU, norm_layer=nn.LayerNorm)
    self.embed = torch.nn.Linear(patch_dim, embed_dim)
    self.fc = nn.Linear(seqlen*embed_dim, num_classes)
    self.loss = torch.nn.CrossEntropyLoss()
    self.reset_parameters()

  def reset_parameters(self):
    init_weights_vit_timm(self)
    
  def forward(self, x):
    # Linear projection
    x = self.embed(x)
            
    # Encoder
    x = self.encoder(x)
    x = x.flatten(start_dim=1)

    # Classification head
    x = self.fc(x)
    return x
    
  def configure_optimizers(self):
    optimizer = Adam(self.parameters(), lr=self.hparams.lr)
    # this decays the learning rate to 0 after max_epochs using cosine annealing
    scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
    return [optimizer], [scheduler]

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss(y_hat, y)
    return loss
    

  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss(y_hat, y)
    acc = accuracy(y_hat, y)
    return {"y_hat": y_hat, "test_loss": loss, "test_acc": acc}

  def test_epoch_end(self, outputs):
    avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
    self.log("test_loss", avg_loss, on_epoch=True, prog_bar=True)
    self.log("test_acc", avg_acc*100., on_epoch=True, prog_bar=True)

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx)

  def validation_epoch_end(self, outputs):
    return self.test_epoch_end(outputs)

def get_args():
    parser = argparse.ArgumentParser()

    # model training hyperparameters
    parser.add_argument('--depth', type=int, default=12, help='depth')
    parser.add_argument('--embed_dim', type=int, default=16, help='embedding dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='num_heads')
    parser.add_argument('--patch_num', type=int, default=8, help='patch_num')
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel size')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: )')
    parser.add_argument('--max-epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0)')
    parser.add_argument("--path", type=str, default="data/speech_commands/")
    parser.add_argument('--accelerator', default='gpu', type=str, metavar='N')
    parser.add_argument('--devices', default=1, type=int, metavar='N')
    parser.add_argument('--dataset', default='cifar10', type=str, metavar='N')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N')
    parser.add_argument("--num-classes", type=int, default=37)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--precision", default=16)
    parser.add_argument("--no-wandb", default=False, action='store_true')
    parser.add_argument("--threshold", type=float, default=0.6)

    args = parser.parse_args("")
    return args


if __name__ == "__main__":
    args = get_args()
    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)
    datamodule = KWSDataModule(batch_size=args.batch_size, num_workers=args.num_workers,
                               path=args.path, n_fft=args.n_fft, n_mels=args.n_mels,
                               win_length=args.win_length, hop_length=args.hop_length,
                               class_dict=CLASS_TO_IDX)
    datamodule.setup()
    data = iter(datamodule.test_dataloader()).next()
    patch_dim = data[0].shape[-1]
    seqlen = data[0].shape[-2]
    print("Embed dim:", args.embed_dim)
    print("Patch size:", 32 // args.patch_num)
    print("Sequence length:", seqlen)
    model = KWSModule(num_classes=37, lr=args.lr, epochs=args.max_epochs, 
                           depth=args.depth, embed_dim=args.embed_dim, head=args.num_heads,
                           patch_dim=patch_dim, seqlen=seqlen,)
    device = torch.device('cpu')
    model.load_state_dict(torch.load('transformers.pth',map_location = device))
    temp = 'temp.wav'

    import sounddevice as sd
    import PySimpleGUI as sg
    import time

    sample_rate = 16000
    sd.default.samplerate = sample_rate
    sd.default.channels = 1
    sg.theme('Dark Amber')
    layout = [
        [sg.Text('Say it!', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 140), key='-OUTPUT-'),],
        [sg.Text('', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 100), key='-STATUS-'),],
        [sg.Text('Speed', expand_x=True, font=("Helvetica", 28), key='-TIME-')],
    ]
    window = sg.Window('KWS Inference', layout, location=(0,0), resizable=True).Finalize()
    window.Maximize()
    window.BringToFront()

    total_runtime = 0
    n_loops = 0
    transforms = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                            n_fft=args.n_fft,
                                            win_length=args.win_length,
                                            hop_length=args.hop_length,
                                            n_mels=args.n_mels,
                                            power=2.0)    
    while True:
        event, values = window.read(100)
        if event == sg.WIN_CLOSED:
            break
        
        waveform = sd.rec(sample_rate).squeeze()
        
        sd.wait()
        if waveform.max() > 1.0:
            continue
        start_time = time.time()
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        mel = ToTensor()(librosa.power_to_db(transforms(waveform).squeeze().numpy(), ref=np.max))
        mel = mel.unsqueeze(0)
        mel = rearrange(mel, 'b c (p1 h) (p2 w) -> b (p1 p2) (c h w)', p1= 16, p2= 8)
        pred = model(mel)
        temp = torch.functional.F.softmax(pred, dim=1)
        max_prob =  temp.max()
        elapsed_time = time.time() - start_time
        total_runtime += elapsed_time
        n_loops += 1
        ave_pred_time = total_runtime / n_loops
        if max_prob > args.threshold:
            pred = int(torch.argmax(pred))
            for key,value in CLASS_TO_IDX.items():
                if value == pred:
                    human_label = key
                    print(human_label)
            window['-OUTPUT-'].update(human_label)
            window['-OUTPUT-'].update(human_label)
            if human_label == "stop":
                window['-STATUS-'].update("Goodbye!")
                # refresh window
                window.refresh()
                time.sleep(1)
                break
                
        else:
            window['-OUTPUT-'].update("...")
        
        window['-TIME-'].update(f"{ave_pred_time:.2f} sec")


    window.close()

