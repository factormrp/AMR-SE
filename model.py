##### MODEL INSTANTIATION FUNCTION (SET HYPERPARAMETERS HERE)
def instantiate(vocab,device):
    ntokens = len(vocab.stoi) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 2048 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8 # the number of heads in the multiheadattention models
    dropout = 0.1 # the dropout value
    return TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

#####################################################################################
##### BASIC TRANSFORMER IMPLEMENTATION FROM PYTORCH WEBSITE
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#####################################################################################
##### BASIC TRAIN AND EVALUATE IMPLEMENTATION FROM PYTORCH WEBSITE
import time

def train(model,sequen_size,device,criterion,optimizer,scheduler):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(sequen_size).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, sequen_size)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != sequen_size:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // sequen_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(model,criterion,eval_model,data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(sequen_size).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, sequen_size):
            data, targets = get_batch(data_source, i)
            if data.size(0) != sequen_size:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

#####################################################################################
##### BASIC BATCHING IMPLEMENTATION FROM PYTORCH WEBSITE
def data_process(raw_text_iter,tokenizer,vocab):
  data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                       dtype=torch.long) for item in raw_text_iter]
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data,batch_size,device):
    # Divide the dataset into batch_size parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)    
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)

def get_batch(source,i,sequen_size):
    seq_len = min(sequen_size, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

#####################################################################################
##### DATA SERVING IMPLEMENTATION
import io
import os
# import torch
from torchtext.utils import extract_archive 
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def concat_training_sets(archive,difficulty):
    if archive:
        print("Extracting data from archive...")
        files = extract_archive(archive)
        for file in files:
            file_process(file,difficulty)
    else:
        print("Walking data folder...")
        for p,d,files in os.walk('./mathematics_dataset-v1.0'):
            for file in files:
                file_process(os.path.join(p,file),difficulty)

def file_process(file,difficulty):
    # create testing master files regardless of difficulty
    if 'interpolate' in file:
        with open(file,'r') as f:
            add_contents('interpolate.txt',f)    
            f.close()

    if 'extrapolate' in file:
        with open(file,'r') as f:
            add_contents('extrapolate.txt',f)    
            f.close()

    # create training file(s), processing only specified difficulty
    if 'train-easy' in file and (difficulty == '-easy' or difficulty == '-all'):
        with open(file,'r') as f:
            add_contents('train{}.txt'.format(difficulty),f)
            f.close()

    if 'train-medium' in file and (difficulty == '-medium' or difficulty == '-all'):
        with open(file,'r') as f:
            add_contents('train{}.txt'.format(difficulty),f)
            f.close()

    if 'train-hard' in file and (difficulty == '-hard' or difficulty == '-all'):
        with open(file,'r') as f:
            add_contents('train{}.txt'.format(difficulty),f)
            f.close()

def add_contents(filename,f):
    # open the file at filename and write every line from f into it
    with open(filename,'a') as w:
        for i in range(4000):
            w.write(f.readline().strip())
        w.close()

def data_serve(device,difficulty='-all'):
    # create training data files in current working directory
    print("Creating Training Data...")
    if os.path.exists('mathematics_dataset-v1.0'):
        concat_training_sets(None,difficulty)
    else:
        archive = './mathematics_dataset-v1.0.tar.gz'
        concat_training_sets(archive,difficulty)

    # set training and testing filepaths
    print("-> setting filepaths to train and test data...")
    trainpath = os.path.abspath('train{}.txt'.format(difficulty))
    interpath = os.path.abspath('interpolate.txt')
    extrapath = os.path.abspath('extrapolate.txt')

    # tokenize the data and create iterator
    print("-> building a vocabulary...")
    tokenizer = get_tokenizer('basic_english')
    lexicon = build_vocab_from_iterator(map(tokenizer,iter(io.open(trainpath,encoding='utf-8'))))

    print("-> converting data into tokens...")
    train_data = data_process(iter(io.open(trainpath, encoding="utf8")),tokenizer=tokenizer,vocab=lexicon)
    val_data = data_process(iter(io.open(interpath, encoding="utf8")),tokenizer=tokenizer,vocab=lexicon)
    test_data = data_process(iter(io.open(extrapath, encoding="utf8")),tokenizer=tokenizer,vocab=lexicon)

    print("-> batching data and returning sets...")
    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_data,batch_size,device)
    val_data = batchify(val_data,eval_batch_size,device)
    test_data = batchify(test_data,eval_batch_size,device)

    print("-> serving data\n.\n.\n.")
    return train_data,val_data,test_data,lexicon

#####################################################################################
##### DRIVER IMPLEMENTATION
def main(args):
    # remove any generated files before creating training data
    import cleandir as clean
    clean.main()
    
    # check for erroneous arguments
    if len(args)>2:
        print("Too many arguments. IMPROPER SCRIPT EXECUTION")
        exit()
    elif len(args)>1 and args[1] != '-easy' and args[1] != '-medium' and args[1] != '-hard': 
        print("Unknown argument. IMPROPER SCRIPT EXECUTION")
        exit()

    # set the device on which to perform computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize data holds
    train_data = None
    val_data = None
    test_data = None
    vocab = None

    # serve data
    if len(args)>1:
        train_data,val_data,test_data,vocab = data_serve(device,''.join(args[1:]))
    else:
        train_data,val_data,test_data,vocab = data_serve(device=device)

    # instantiate the model
    print("Setting up the model...")
    model = instantiate(vocab,device)

    print("-> choosing criterion...")
    criterion = nn.CrossEntropyLoss()
    print("-> choosing optimizer\n.\n.\n.")
    lr = 0.0001 # learning rate
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.995),eps=10**-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")
    epochs = 3 # The number of epochs
    best_model = None
    sequen_size = 160 # The maximum size of each sequence

    print("Training the model...")
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model,sequen_size,device,criterion,optimizer,scheduler)
        val_loss = evaluate(model,criterion,model,val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

    print("Evaluating the model...")
    test_loss = evaluate(model,criterion,best_model,test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

if __name__ == "__main__":
    import sys
    main(sys.argv)
