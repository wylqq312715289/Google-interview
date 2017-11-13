import os
import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from data_utils import Dictionary, Corpus

# Hyper Parameters
embed_size = 128  # 是语言模型中embedding层的维度
hidden_size = 1024 # 隐藏层大小
num_layers = 2 # lstm中层数
seq_length = 40 # 训练时句子长度
num_epochs = 65 # 训练模型的轮数
num_samples = 120 # 最后生成的样本长度
batch_size = 40  # 分批训练的大小
learning_rate = 0.002 #模型训练时的学习率

use_gpu = torch.cuda.is_available() #查看GPU是否可用
# Load Penn Treebank Dataset
train_path = './data/train.txt' # 训练语料文件存放位置
sample_path = './cache/sample.txt' # 生成文件存放位置 

# RNN Based Language Model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.GRU(embed_size, hidden_size, num_layers, dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x) 
        # Forward propagate RNN  
        out, h = self.lstm(x, h)
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        # Decode hidden states of all time step
        out = self.linear(out)  
        return out, h

# Truncated Backpropagation 截断以后的网络不求梯度
def detach(states): return [state.detach() for state in states] 

# Training
def train_model(model, ids, criterion, optimizer):
    # 重新训练模型时注释掉该行
    # if os.path.exists('./checkpoints/model_60.pth'): os.remove('./checkpoints/model_60.pth')
    if os.path.exists('./checkpoints/model_60.pth'):
        model.load_state_dict(torch.load('./checkpoints/model_60.pth'))
        return model
    for epoch in range(num_epochs):
        # Initial hidden and memory states
        if use_gpu:
            states = (
                Variable(torch.zeros(num_layers, batch_size, hidden_size)).cuda(),
                Variable(torch.zeros(num_layers, batch_size, hidden_size)).cuda(),
            )
        else:
            states = (
                Variable(torch.zeros(num_layers, batch_size, hidden_size)),
                Variable(torch.zeros(num_layers, batch_size, hidden_size)),
            )
        for i in range(0, ids.size(1) - seq_length, seq_length):
            # Get batch inputs and targets
            inputs = Variable(ids[:, i:i+seq_length])
            targets = Variable(ids[:, (i+1):(i+1)+seq_length].contiguous())
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
            # Forward + Backward + Optimize
            model.zero_grad()
            states = detach(states)
            outputs, states = model(inputs, states) 
            loss = criterion(outputs, targets.view(-1)) # 计算梯度
            loss.backward() # 反向传播
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5) # 对网络进行梯度裁剪，因为RNN中容易出现梯度爆炸的问题
            optimizer.step()
            step = (i+1) // seq_length
            if step % 100 == 0:
                print ('Epoch [%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                       (epoch+1, num_epochs, loss.data[0], np.exp(loss.data[0])))
            if (epoch + 1) % 20 == 0:
                if not os.path.exists('./checkpoints'): os.mkdir('./checkpoints/')
                torch.save(model.state_dict(),'./checkpoints/model_{}.pth'.format(epoch+ 1))
    return model

# Sampling
def sampling(model,vocab_size,corpus):
    with open(sample_path, 'w') as f:
        # Set intial hidden ane memory states
        if use_gpu:
            state = (
                Variable(torch.zeros(num_layers, 1, hidden_size)).cuda(),
                Variable(torch.zeros(num_layers, 1, hidden_size)).cuda(),
            )
        else:
            state = (
                Variable(torch.zeros(num_layers, 1, hidden_size)),
                Variable(torch.zeros(num_layers, 1, hidden_size)),
            )
        # Select one word id randomly
        prob = torch.ones(vocab_size)
        # 随机生成一个1*1的input
        input = Variable(torch.multinomial(prob, num_samples=1).unsqueeze(1),volatile=True)
        if use_gpu: input = input.cuda()
        for i in range(num_samples):
            # Forward propagate rnn 
            output, state = model(input, state)   
            # Sample a word id
            prob = output.squeeze().data.exp().cpu()
            word_id = torch.multinomial(prob, 1)[0]
            # Feed sampled word id to next time step
            input.data.fill_(word_id)
            # File write
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)
            if (i+1) % 100 == 0:
                print('Sampled [%d/%d] words and save to %s'%(i+1, num_samples, sample_path))

def main():
    corpus = Corpus() # 构建语料库类
    ids = corpus.get_data(train_path, batch_size) # 返回数据矩阵 ids[i,j]表示 单词的索引下标
    # ids.shape = 20*889
    vocab_size = len(corpus.dictionary)
    print("vocab_size=",vocab_size) # 查看语料中字典库的大小
    num_batches = ids.size(1) // seq_length
    model = RNNLM(vocab_size, embed_size, hidden_size, num_layers)
    if use_gpu: model.cuda()
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = train_model(model, ids, criterion, optimizer)
    sampling(model, vocab_size, corpus)

if __name__ == '__main__':
    main()














