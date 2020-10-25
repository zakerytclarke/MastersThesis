"""
Code By Hrituraj Singh
"""
import operator
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from Embeddings import UnaryEmbedding, BinaryEmbedding
from Embeddings import VariableEmbedding, TypesEmbedding, PointerEmbedding



# Default word tokens
PAD_token = 0  # Used for padding short sentences
UNK_token = 1  # Unknown or OOV word
SOS_token = 2  # Start of the sequence
EOS_token = 3  # End of the sequence


class FOLDecoderBase(nn.Module):
    """
    FOLDecoder for the seq to seq model. This encoder is LSTM based only. Not
    recommended to be overridden with BERT/CNNS2S. Create new one instead
    """

    def __init__(self, vocabSizes = [3000, 3000, 3000, 3000, 3000], embeddingSizes = [100, 100, 50], encoderHiddenSize = 200, numLayers = 1, dropout = 0.1, pretrained = False, pretrain = None, batchSize = 32, maxSeqLenOut = 30):
        """
        Initializer for the FOLDecoderBase class

        [Inputs]
        vocabSizes: List of the vocab sizes in order: unary, binary, variables, pointers, Types
        embeddingSizes: List of the Embedding sizes in order: unary, binary, pointers
        encoderHiddenSize: Hidden size of the encoder
        maxLen: Maximum length of the sequence
        numLayers: Number of layers in the LSTM (defaults to 1)
        dropout: Dropout rate for the network
        pretrained: if pretrained embeddings need to be loaded
        pretrain: list Weights of the pretrained embeddings, if pretrained = True, must be passed as arg order: unary, binary
        batchSize: batch size that will be used for this decoder instance
        maxSeqLenOut: Maximum Output Sequence Length
        """
        super(FOLDecoderBase, self).__init__()
        
        # LSTM Details
        self.numLayers = numLayers
        self.hiddenSize = 2 * encoderHiddenSize # Since we are using unidirectional decoder while encoder is bidirectional


        # Getting the sizes of the vocabularies
        self.unaryVocabSize = vocabSizes[0]
        self.binaryVocabSize = vocabSizes[1]
        self.variablesVocabSize = vocabSizes[2]
        self.pointersVocabSize = vocabSizes[3]
        self.typesVocabularySize = vocabSizes[4]

        # Getting the sizes of the embeddings
        self.unaryEmbeddingSize = embeddingSizes[0]
        self.binaryEmbeddingSize = embeddingSizes[1]
        self.pointersEmbeddingSize = embeddingSizes[2]

        # Getting the embedding functions
        self.unaryEmbedding = UnaryEmbedding(self.unaryEmbeddingSize, self.unaryVocabSize)
        self.binaryEmbedding = BinaryEmbedding(self.binaryEmbeddingSize, self.binaryVocabSize)
        self.variableEmbedding = VariableEmbedding(self.variablesVocabSize)
        self.pointerEmbedding = PointerEmbedding(self.pointersEmbeddingSize, self.pointersVocabSize)
        self.typesEmbedding = TypesEmbedding(self.typesVocabularySize)

        # Getting the linear heads to be used on top of LSTM
        self.unary = nn.Linear(self.hiddenSize, self.unaryVocabSize)
        self.binary = nn.Linear(self.hiddenSize, self.binaryVocabSize)
        self.variables = nn.Linear(self.hiddenSize, self.variablesVocabSize)
        self.pointers = nn.Linear(self.hiddenSize, self.pointersVocabSize)
        self.types = nn.Linear(self.hiddenSize, self.typesVocabularySize)

        # Size of the concatenated input embedding for decoder
        self.embeddingSize = self.unaryEmbeddingSize + self.binaryEmbeddingSize + self.variablesVocabSize + self.pointersEmbeddingSize + self.typesVocabularySize + encoderHiddenSize*2

        if pretrained: # Deprecated
            self.unary.embeddings.weights = pretrain[0]
            self.binary.embeddings.weights = pretrain[1]
            self.pointers.embeddings.weights = pretrain[2]
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.embeddingSize, self.hiddenSize, num_layers=numLayers, bidirectional = False, batch_first = True)

        # Copy Mechanism
        self.copyProjection = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.copyDecisions = nn.Linear(self.hiddenSize, 1)
        self.sigmoid = nn.Sigmoid() # initialize sigmoid layer



    def encDecAttention(self, query, key, value):
        """
        Applying the encoder-decoder attention

        [Input]
        query: Query to be used for attention. 
        key: Keys to be accessed using query
        value: value to be used weighted by attention weights

        [Output]
        context: Context vector after applying attention weights
        """

        attnWeights = F.softmax(torch.bmm(query, key), dim=-1)
        context = torch.bmm(attnWeights, value)
        return context

    def decAttention(self, query, key, value):
        """
        Applying self attention over already predicted 
        decoder outputs

        [Input]
        query: Query for the current step
        key: Keys to be accessed using query
        value: value to be used weighted by attention weights

        [Output]
        context: Context vector for the step
        """
        if len(key) == 0:
            return torch.zeros_like(query)

        attnWeights = F.softmax(torch.bmm(query, key), dim=-1)

        context = torch.bmm(attnWeights, value)
        return context

    def copyMechanism(self, query, key, value):
        """
        Applying Copy Mechanism on the decoder past

        [Input]
        query: Query for the current step
        key: Keys to be accessed using query
        value: value to be used weighted by attention weights

        [Output]
        context: context vector to be copied from
        weights: copy weights to decide where to copy from
        """
        if len(key) == 0:
            return torch.zeros_like(query), torch.zeros((len(query),1)).cuda()


        query = self.copyProjection(query)
        key = key.transpose(1,2)
        key = self.copyProjection(key)
        key = key.transpose(1,2)

        attnLogits = torch.bmm(query, key)
        attnWeights = F.softmax(attnLogits, dim=-1)



        context = torch.bmm(attnWeights, value)


        return context, attnLogits.squeeze(1)


    def forward(self, variables, hidden, encoder, past):
        """
        Performs the feed forwards on LSTM

        [Input]
        variables: a tensor/list of variables in order: unary, binary, variables, types
        hidden: initializing hidden state for the decoder
        encoder: hidden states output from the encoder
        past: Hidden states which the decoder has already decoded in the past
        """


        unary = self.unaryEmbedding(variables[:,:,0].long())
        binary = self.binaryEmbedding(variables[:,:,1].long())
        variables = self.variableEmbedding(variables[:,:,2].long())
        pointers = self.pointerEmbedding(variables[:,:,3].long())
        types = self.typesEmbedding(variables[:,:,4].long())


        # Setting up Attention
        encDecKey = encoder.transpose(1,2)
        encDecQuery = hidden[0].transpose(0,1) # Bringing batchsize to first position
        encDecValue = encoder

        decKey = torch.cat(past, dim=1).transpose(1,2) if len(past) else torch.Tensor(past).cuda()
        decQuery = hidden[0].transpose(0,1)
        decValue = torch.cat(past, dim=1) if len(past) else torch.Tensor(past)


        # Applying the attention (both encoder-decoder as well as decoder)
        encoderContext = self.encDecAttention(encDecQuery, encDecKey, encDecValue)
#       decoderContext = self.decAttention(decQuery, decKey, decValue)

        #concatenated embedding
        embedding = torch.cat((unary, binary, variables, pointers, types, encoderContext), dim = -1)

        output, hidden = self.lstm(embedding, hidden)

        # Getting the outputs using the heads
        unaryOutput = self.unary(output)
        binaryOutput = self.binary(output)
        pointersOutput = self.pointers(output)
        typesOutput = self.types(output)

        # Getting the variable output using copy mechanism
        copyDecisions = self.copyDecisions(output)
        copyDecisions = self.sigmoid(copyDecisions).squeeze(1)
        copyContext, copyWeights = self.copyMechanism(decQuery, decKey, decValue)
        output = copyDecisions * copyContext.squeeze(1) + (1 - copyDecisions) * output.squeeze(1)
        output = output.unsqueeze(1)
        variablesOutput = self.variables(output)


        output = [unaryOutput, binaryOutput, variablesOutput, pointersOutput, typesOutput]
        copy = [copyWeights, copyDecisions]

        return output, hidden, copy