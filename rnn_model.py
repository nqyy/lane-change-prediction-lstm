import numpy
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


#load data into pytorch tensor
input_data = []
for i in range(1, 11):
    idx_str = '{0:02}'.format(i)
    pickle_in = open("output/result"+idx_str+".pickle","rb")
    temp_data = pickle.load(pickle_in)
    print ("Loaded "+idx_str+" data pack")
    input_data.extend(temp_data)

# pickle_in = open("result.pickle","rb")
# input_data = pickle.load(pickle_in)
random.shuffle(input_data)
print(len(input_data))
size = len(input_data)
training_set = input_data[:int(size*0.8)]
print(len(training_set[0]))
testing_set = input_data[int(size*0.8):]

state_dim = 16
states_per_sequence = 25
hidden_dim = 256
output_size = 2
num_epoch = 10

class PredicterRNN(nn.Module):
    def __init__(self, input_dim=state_dim*states_per_sequence, hidden_dim=256, output_size=2, batch_size=states_per_sequence, num_layers=1):
        super(PredicterRNN, self).__init__()
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)

        # The linear layer that maps from hidden state space to output space
        self.dense1 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.dense2 = nn.Linear(int(hidden_dim/2), output_size)
    
    # def init_hidden(self):
    #     # This is what we'll initialise our hidden state as
    #     return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
    #             torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        input = input.view(-1, 25, 16)
        #print(input.size())
        lstm_out, _ = self.lstm(input)#.view(state_dim, self.batch_size, -1))
        output_space = self.dense1(lstm_out.view(self.batch_size, -1))
        output_space = torch.tanh(output_space)
        output_space = self.dense2(output_space)
        output_space = torch.tanh(output_space)
        output_scores = F.log_softmax(output_space, dim=1)
        return output_scores


model = PredicterRNN(state_dim, hidden_dim, output_size, states_per_sequence)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


#load the data into pytorch
def prepare_sequence(state_sequence, tag):
    s_seq_tensor = torch.tensor(state_sequence, dtype=torch.float)
    #print(s_seq_tensor, s_seq_tensor.size())
    tag_list = [tag] * len(state_sequence)
    labels = torch.tensor(tag_list, dtype=torch.long)
    #print(labels.size())
    return s_seq_tensor, labels


for epoch in range(num_epoch):
    random.shuffle(training_set)
    for state_sequence, tag in training_set:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of state sequences.
        sentence_in, label = prepare_sequence(state_sequence, tag)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, label)
        loss.backward()
        optimizer.step()

    # # See what the scores are after training
    correct, total = 0, 0
    with torch.no_grad():
        for state_sequence, tag in testing_set:
            sentence_in, label = prepare_sequence(state_sequence, tag)
            output_scores = model(sentence_in)
            val, idx = output_scores[-1].max(0)
            if idx == tag:
                correct += 1
            total += 1

        print("accuracy:", correct/total)