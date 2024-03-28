import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
#from torchvision.models import AlexNet
from torchvision.utils import _log_api_usage_once
##################################LeNet###################################
class LeNet(nn.Module):
    def __init__(self,classes):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.f1=nn.Linear(16*5*5,120)
        self.f2=nn.Linear(120,84)
        self.f3=nn.Linear(84,classes)

    def __setattr__(self, key, value):
        if isinstance(value,nn.Module):
            self._modules[key]=value

    def forward(self,x):
        out=self.conv1(x)
        out=F.relu(out)
        out=F.max_pool2d(out,2)
        out=self.conv2(out)
        out=F.relu(out)
        out=F.max_pool2d(out,2)
        out=out.view(out.size(0),-1)
        out=self.f1(out)
        out=F.relu(out)
        out=self.f2(out)
        out=F.relu(out)
        out=self.f3(out)
        return out

    __call__=forward

    def initialize(self):
        for module in self._modules.values():
            nn.init.kaiming_normal_(module.weight.data)
            if  module.bias is not None:
                module.bias.data.zero_()


##################################Sequential###################################

class LeNetSequential(nn.Module):
    def __init__(self,classes):
        super(LeNetSequential,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.classifier=nn.Sequential(
            nn.Linear(400,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,classes),
        )

    def forward(self,x):
        out=self.features(x)
        out.view(out.size[0],-1)
        out=self.classifier(out)
        return out
# flag=True
flag=False
if flag:
    net=LeNetSequential(classes=2)
    test=torch.randn((4, 3, 32, 32), dtype=torch.float32)
    # print(net._modules)
    # print(net._modules['features'])
    # print(net.features)
    print(net.features._modules)  #差别就在features的_modules里，存储的Module子类的有序字典是以数值还是以具体名字命名
    print(net.features._modules['0'])

##################################OrderedSequential###################################
class LeNetOrderedSequential(nn.Module):
    def __init__(self, classes):
        super(LeNetOrderedSequential, self).__init__()
        self.features = nn.Sequential(OrderedDict({
            'conv1':nn.Conv2d(3, 6, 5),
            'relu1':nn.ReLU(),
            'pool1':nn.MaxPool2d(kernel_size=2, stride=2),
            'conv2':nn.Conv2d(6, 16, 5),
            'relu2':nn.ReLU(),
            'pool2':nn.MaxPool2d(kernel_size=2, stride=2),
        }))

        self.classifier = nn.Sequential(OrderedDict({
            'linear1':nn.Linear(400, 120),
            'relu1':nn.ReLU(),
            'linear2':nn.Linear(120, 84),
            'relu2':nn.ReLU(),
            'linear3':nn.Linear(84, classes),
        }))


    def forward(self, x):
        out = self.features(x)
        out.view(out.size[0], -1)
        out = self.classifier(out)
        return out

# flag=True
flag=False
if flag:
    net=LeNetOrderedSequential(classes=2)
    test=torch.randn((4, 3, 32, 32), dtype=torch.float32)
    # print(net._modules)
    # print(net._modules['features'])
    # print(net.features)
    print(net.features._modules)
    print(net.features._modules['conv2'])
    # print(net.features[0])

##################################ModelList###################################
class ModuleList(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.ModuleList([nn.Linear(10,10) for _ in range(20)])
    def forward(self,x):
        for linear in self.linear:
            x=linear(x)
        return x

# flag=True
flag=False
if flag:
    net=ModuleList()
    test=torch.eye(10,10)
    print(net(test))

##################################ModuleDict###################################

class ModuleDict(nn.Module):
    def __init__(self,classes):
        super(ModuleDict, self).__init__()
        self.features=nn.ModuleDict({
            'conv1': nn.Conv2d(3, 6, 5),
            'relu1': nn.ReLU(),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),
            'conv2': nn.Conv2d(6, 16, 5),
            'relu2': nn.ReLU(),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
        })

        self.classfier=nn.ModuleDict({
            'linear1': nn.Linear(400, 120),
            'relu1': nn.ReLU(),
            'linear2': nn.Linear(120, 84),
            'relu2': nn.ReLU(),
            'linear3': nn.Linear(84, classes),
        })


    def forward(self,out,*feature_index,flag=True,**classifier_index):
        if flag:
            for i in feature_index:
                out=self.features[i](out)
            out=out.view(out.size(0), -1)
            for j in classifier_index.values():
                out=self.classfier[j](out)
            return out

# flag=True
flag=False
if flag :
    net=ModuleDict(classes=2)
    test=torch.randn((4,3,32,32))
    res=net(test,'conv1','relu1','pool1','conv2','relu2','pool2',c1='linear1',c2='relu1',c3='linear2',c4='relu2',c5='linear3')
    print(res)


##################################AlexNet###################################
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = nn.Sequential(OrderedDict({
            'conv1':nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            'relu1':nn.ReLU(inplace=True),
            'pool1':nn.MaxPool2d(kernel_size=3, stride=2),

            'conv2':nn.Conv2d(64, 192, kernel_size=5, padding=2),
            'relu2':nn.ReLU(inplace=True),
            'pool2':nn.MaxPool2d(kernel_size=3, stride=2),

            'conv3':nn.Conv2d(192, 384, kernel_size=3, padding=1),
            'relu3':nn.ReLU(inplace=True),

            'conv4':nn.Conv2d(384, 256, kernel_size=3, padding=1),
            'relu4':nn.ReLU(inplace=True),

            'conv5':nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'relu5':nn.ReLU(inplace=True),
            'pool5':nn.MaxPool2d(kernel_size=3, stride=2),
        })

        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
# flag=True
flag=False
net=AlexNet(num_classes=100,dropout=0.5)
#print(net._modules['features']._modules.keys())
#odict_keys(['conv1', 'relu1', 'pool1', 'conv2', 'relu2', 'pool2', 'conv3', 'relu3', 'conv4', 'relu4', 'conv5', 'relu5', 'pool5'])

#------------------------------------Generator------------------------------------
class Generator(nn.Module):
    def __init__(self,sf=100,mf=128,ef=3):  #start feature,middle feature, end feature
        super(Generator,self).__init__() #B 100 1 1--->B 3 64 64
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=sf,out_channels=mf*8,kernel_size=(4,4),stride=(1,1),padding=(0,0)),
            nn.BatchNorm2d(num_features=mf*8),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=mf*8,out_channels=mf*4,kernel_size=(4,4),stride=(2,2),padding=(1,1)),
            nn.BatchNorm2d(num_features=mf*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=mf * 4, out_channels=mf * 2, kernel_size=(4, 4), stride=(2, 2),padding=(1, 1)),
            nn.BatchNorm2d(num_features=mf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=mf * 2, out_channels=mf * 1, kernel_size=(4, 4), stride=(2, 2),padding=(1, 1)),
            nn.BatchNorm2d(num_features=mf * 1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=mf, out_channels=ef, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
            
        )

    def forward(self,x):
        return self.main(x)

    def initailize(self,w_mean=0., w_std=0.02, b_mean=1, b_std=0.02):#initialize weight for Conv weight and BatchNorm weight
        for m in self.modules():
            cur_classname = m.__class__.__name__
            if 'Conv' in cur_classname:
                nn.init.normal_(m.weight.data,w_mean,w_std)
                nn.init.constant_(m.bias.data,0)
            if 'Batch' in cur_classname:
                nn.init.normal_(m.weight.data,b_mean,b_std)
                nn.init.constant_(m.bias.data,0)

# flag = 1
flag = 0
if flag :
    from torchsummary import summary
    g = Generator()
    print(summary(g,(100,1,1),device='cpu'))


#------------------------------------Discriminator------------------------------------
class Discriminator(nn.Module):
    def __init__(self, sf=3, mf=128, ef=100):  # start feature,middle feature, end feature
        super(Discriminator, self).__init__()  # B 3 64 64--->B 100 1 1
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=sf, out_channels=mf, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=mf),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=mf, out_channels=mf * 2, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1)),
            nn.BatchNorm2d(num_features=mf * 2),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=mf * 2, out_channels=mf * 4, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1)),
            nn.BatchNorm2d(num_features=mf * 4),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=mf * 4, out_channels=mf * 8, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1)),
            nn.BatchNorm2d(num_features=mf * 8),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=mf*8, out_channels=ef, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            nn.Sigmoid()

        )

    def forward(self, x):
        return self.main(x)

    def initailize(self, w_mean=0., w_std=0.02, b_mean=1,
                   b_std=0.02):  # initialize weight for Conv weight and BatchNorm weight
        for m in self.modules():
            cur_classname = m.__class__.__name__
            if 'Conv' in cur_classname:
                nn.init.normal_(m.weight.data, w_mean, w_std)
                nn.init.constant_(m.bias.data, 0)
            if 'Batch' in cur_classname:
                nn.init.normal_(m.weight.data, b_mean, b_std)
                nn.init.constant_(m.bias.data, 0)

# flag = 1
flag = 0
if flag:
    from torchsummary import summary

    d = Discriminator()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d.to(device)
    print(summary(d, (3, 64,64)))

#------------------------------------RCNN----------------------------------------
class RNN(nn.Module):
    def __init__(self,in_num,hi_num,out_num):
        super(RNN,self).__init__()
        self.lx = nn.Linear(in_num,hi_num)
        self.lh_t = nn.Linear(hi_num,out_num)
        self.lh_t_1  =nn.Linear(hi_num,hi_num)

    def forward(self,xt,ht_1):
        ht = self.lx(xt)
        ht = nn.Tanh()(ht+self.lh_t_1(ht_1))
        ot = nn.Softmax(dim=1)(self.lh_t(ht))
        return ot,ht

    def initialize_h0(self):
        return torch.zeros((128,))

# flag = 1
flag = 0
if flag:
    rnn = RNN(57,128,17)
    h0 = rnn.initialize_h0()
    x1 = torch.ones((1,57))
    print(rnn(x1,h0)[0])

#Unet
class Unet(nn.Module):
    def __init__(self, in_features=3, out_features=1, base_features=64):
        super(Unet, self).__init__()
        self.encoder1 = self._block(in_features, base_features, 'encoder1')
        self.encoder2 = self._block(base_features, base_features * 2, 'encoder2')
        self.encoder3 = self._block(base_features * 2, base_features * 4, 'encoder3')
        self.encoder4 = self._block(base_features * 4, base_features * 8, 'encoder4')

        self.bottleneck = self._block(base_features * 8, base_features * 16, 'bottleneck')

        self.decoder4 = self._block(base_features * 16, base_features * 8, 'decoder4')
        self.decoder3 = self._block(base_features * 8, base_features * 4, 'decoder3')
        self.decoder2 = self._block(base_features * 4, base_features * 2, 'decoder2')
        self.decoder1 = self._block(base_features * 2, base_features * 1, 'decoder1')

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.upconv4 = self.upconv(base_features * 16, base_features * 8)
        self.upconv3 = self.upconv(base_features * 8, base_features * 4)
        self.upconv2 = self.upconv(base_features * 4, base_features * 2)
        self.upconv1 = self.upconv(base_features * 2, base_features * 1)
        self.conv = nn.Conv2d(base_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


    def forward(self,x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.conv(dec1)
        out = nn.Sigmoid()(out)
        return out

    @staticmethod
    def _block(in_channels, out_channels, name):
        return nn.Sequential(
            OrderedDict({
                name+'conv1': nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                name+'bn1': nn.BatchNorm2d(num_features=out_channels),
                name+'relu1': nn.ReLU(inplace=True),
                name+'conv2': nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                name+'bn2': nn.BatchNorm2d(num_features=out_channels),
                name+'relu1': nn.ReLU(inplace=True)
            })
        )

    @staticmethod
    def upconv(in_channels, out_channels):
         return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 2), stride=(2, 2))

# flag = 1
flag = 0
if flag:
    from torchsummary import summary
    model = Unet()
    print(summary(model, (3, 512, 512), batch_size=2, device='cpu'))





