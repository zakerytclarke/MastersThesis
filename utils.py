"""
Code By Hrituraj Singh
"""
import os,json, re
import configparser
import torch
import numpy as np

def cleanText(text, lower=False):
    """
    Cleaner for the text
    """
    if lower:
        text = text.lower()

    text=re.sub(r'(\d+)',r'',text)
    text=text.replace(u',','')
    text=text.replace(u'"','')
    text=text.replace(u'(','')
    text=text.replace(u')','')
    text=text.replace(u'"','')
    text=text.replace(u':','')
    text=text.replace(u"'",'')
    text=text.replace(u"‘‘",'')
    text=text.replace(u"’’",'')
    text=text.replace(u"''",'')
    text=text.replace(u".",'')
    text=text.replace(u"?",'')
    text=text.replace(u"।",'')

    return text


def map_brackets(fol):
    scratch_list = []
    bracket_mapping = {}
    for index,char in enumerate(fol):
        if char == '(':
            scratch_list.append(index)
        elif char == ')':
            last_index = scratch_list[-1]
            bracket_mapping[last_index] = index
            scratch_list = scratch_list[:-1]
    return bracket_mapping

def parsePredicates(test_string):
    """
    Parsing the predicates from the raw text file
    """
    bracket_mapping=map_brackets(test_string)
    
    binary_predicate_pattern = '(\(|,)([A-Za-z0-9_]*)(\([A-Z][0-9]*,[A-Z][0-9]*\))'
    unary_predicate_pattern = '(\(|,)([A-Za-z0-9_]*)(\([A-Z][0-9]*\))'
    functional_pred_form = '[A-Za-z0-9_]*\('
    
    resultBinary = re.finditer(binary_predicate_pattern, test_string)
    resultUnary = re.finditer(unary_predicate_pattern, test_string)
    resultFunctional = re.finditer(functional_pred_form, test_string)
    
    resultBinarypat = re.findall(binary_predicate_pattern, test_string)
    resultUnarypat = re.findall(unary_predicate_pattern, test_string)
    resultFunctionalpat = re.findall(functional_pred_form, test_string)

    
    index_actual_mapping_unary = {}
    ulist = []; blist = []; flist = []
    for index,ele in enumerate(resultUnary):
        start = ele.start()
        end = ele.end()
        current_string = test_string[start:end]
        index_actual_mapping_unary[index] = list(filter(lambda x : x[0]+x[1]+x[2] == current_string, resultUnarypat))[0]
        ulist.append([ele.start(),ele.end()])        
    
    index_actual_mapping_binary = {}
    for index,ele in enumerate(resultBinary):
        start = ele.start()
        end = ele.end()
        current_string = test_string[start:end]
        index_actual_mapping_binary[index] = list(filter(lambda x : x[0]+x[1]+x[2] == current_string, resultBinarypat))[0]
        blist.append([ele.start(),ele.end()])
    
    resultBinarypatMatch = list(map(lambda x : x[1], resultBinarypat))
    resultUnarypatMatch = list(map(lambda x : x[1], resultUnarypat))
    resultFunctionalpatMatch = list(map(lambda x : x[:-1], resultFunctionalpat))
    final_fp = list(filter(lambda x : x not in resultBinarypatMatch and x not in resultUnarypatMatch and len(x) > 0, resultFunctionalpatMatch))

    index_actual_mapping_functional = {}
    count=-1
    for index,ele in enumerate(resultFunctional):
        start = ele.start()
        end = ele.end()
        current_string = test_string[start:end]
        match_list = list(filter(lambda x : x == current_string[:-1], final_fp))
        if len(match_list) == 0:
            continue
        else:
            count+=1
            index_actual_mapping_functional[count] = match_list[0]
        flist.append([ele.start(),ele.end()])

    function_scope_mapping = {}
    for new_index,functional_p in enumerate(flist):
        start=functional_p[0]
        end=functional_p[1]
        functional_scope = bracket_mapping[end-1]
        function_scope_mapping[new_index] = [start,functional_scope]

    index_unary_scope_map = {}
    for index_unary in index_actual_mapping_unary:
        start,end = ulist[index_unary]
        possible_scopes = list(filter(lambda x : start > function_scope_mapping[x][0] and end-1 < function_scope_mapping[x][1] and 'some' not in test_string[flist[x][0]:flist[x][1]] 
                                      and 'and' not in test_string[flist[x][0]:flist[x][1]], list(range(len(flist)))))
        selected_scope = sorted(possible_scopes,key=lambda x : function_scope_mapping[x][1]-function_scope_mapping[x][0])[0]
        index_unary_scope_map[index_unary] = selected_scope
    
    index_binary_scope_map = {}
    for index_binary in index_actual_mapping_binary:
        start,end = blist[index_binary]
        possible_scopes = list(filter(lambda x : start > function_scope_mapping[x][0] and end-1 < function_scope_mapping[x][1] and 'some' not in test_string[flist[x][0]:flist[x][1]] 
                                      and 'and' not in test_string[flist[x][0]:flist[x][1]], list(range(len(flist)))))
        
        selected_scope = sorted(possible_scopes,key=lambda x : function_scope_mapping[x][1]-function_scope_mapping[x][0])[0]
        index_binary_scope_map[index_binary] = selected_scope
    
    index_functional_scope_map = {}
    for index_functional in index_actual_mapping_functional:
        start,end = flist[index_functional]
        possible_scopes = list(filter(lambda x : start > function_scope_mapping[x][0] and end-1 < function_scope_mapping[x][1] and 'some' not in test_string[flist[x][0]:flist[x][1]] 
                                      and 'and' not in test_string[flist[x][0]:flist[x][1]], list(range(len(flist)))))
        if possible_scopes == []:
            selected_scope = -1
            start_index=index_functional
        else:
            selected_scope = sorted(possible_scopes,key=lambda x : function_scope_mapping[x][1]-function_scope_mapping[x][0])[0]
        index_functional_scope_map[index_functional] = selected_scope
    
    functional_scope_elements_mapping_unary = {}
    for functional_scope_index in function_scope_mapping:
        unary_list = list(filter(lambda x : index_unary_scope_map[x] == functional_scope_index, list(index_unary_scope_map.keys())))
        functional_scope_elements_mapping_unary[functional_scope_index] = unary_list

    functional_scope_elements_mapping_binary = {}
    for functional_scope_index in function_scope_mapping:
        binary_list = list(filter(lambda x : index_binary_scope_map[x] == functional_scope_index, list(index_binary_scope_map.keys())))
        functional_scope_elements_mapping_binary[functional_scope_index] = binary_list

    functional_scope_elements_mapping_functional = {}
    for functional_scope_index in function_scope_mapping:
        functional_list = list(filter(lambda x : index_functional_scope_map[x] == functional_scope_index, list(index_functional_scope_map.keys())))
        functional_scope_elements_mapping_functional[functional_scope_index] = functional_list
    
    
    def expand_sequence(index):
        current_unary = functional_scope_elements_mapping_unary[index]
        current_binary = functional_scope_elements_mapping_binary[index]
        current_functional = functional_scope_elements_mapping_functional[index]
        sequence = [test_string[flist[index][0]:flist[index][1]]]
        for unary_index in current_unary:
            sequence.append(index_actual_mapping_unary[unary_index])
        for binary_index in current_binary:
            sequence.append(index_actual_mapping_binary[binary_index])
        for functional_index in current_functional:
            sequence+=expand_sequence(functional_index)
        sequence.append(')')
        return sequence

    leave_current = False
    final_seq_ele = []
    for ele in expand_sequence(start_index):
        if ele == 'some(' or ele == 'and(':
            leave_current = True
            continue
        else:
            if leave_current:
                leave_current = False
                continue
        final_seq_ele.append(ele)


    return final_seq_ele





def parseConfig(filename):
    """
    Parsing the config file
    """
    config = configparser.ConfigParser()
    config.read(filename)
    output = {}
    for section in config.sections():
        output[section] = {}
        for key in config[section]:
            val_str = str(config[section][key])
            if(len(val_str)>0):
                val = parse_value_from_string(val_str) 
            else:
                val = None
            print(section, key,val_str, val)
            output[section][key] = val
    return output



def parse_value_from_string(val_str):
    if(is_int(val_str)):
        val = int(val_str)
    elif(is_float(val_str)):
        val = float(val_str)
    elif(is_list(val_str)):
        val = parse_list(val_str)
    elif(is_bool(val_str)):
        val = parse_bool(val_str)
    else:
        val = val_str
    return val

def is_int(val_str):
    start_digit = 0
    if(val_str[0] =='-'):
        start_digit = 1
    flag = True
    for i in range(start_digit, len(val_str)):
        if(str(val_str[i]) < '0' or str(val_str[i]) > '9'):
            flag = False
            break
    return flag

def is_float(val_str):
    flag = False
    if('.' in val_str and len(val_str.split('.'))==2):
        if(is_int(val_str.split('.')[0]) and is_int(val_str.split('.')[1])):
            flag = True
        else:
            flag = False
    elif('e' in val_str and len(val_str.split('e'))==2):
        if(is_int(val_str.split('e')[0]) and is_int(val_str.split('e')[1])):
            flag = True
        else:
            flag = False       
    else:
        flag = False
    return flag 

def is_bool(var_str):
    if( var_str=='True' or var_str == 'true' or var_str =='False' or var_str=='false'):
        return True
    else:
        return False
    
def parse_bool(var_str):
    if(var_str=='True' or var_str == 'true' ):
        return True
    else:
        return False
     
def is_list(val_str):
    if(val_str[0] == '[' and val_str[-1] == ']'):
        return True
    else:
        return False
    
def parse_list(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if(is_int(item)):
            output.append(int(item))
        elif(is_float(item)):
            output.append(float(item))
        elif(is_bool(item)):
            output.append(parse_bool(item))
        else:
            output.append(item)
    return output


def loadGlove(path, vocab, pretrainedembeddingSize):
    """
    loads the Glove Embedding
    [Input]
    path: Path to the Glove Embeddings file
    vocab: vocabulary for which these need to be loaded
    pretrainedembeddingSize: Pretrained embedding size, Keep this same as your model embedding size

    [Output]
    EmbWeights: Embedding weight matrix as torch Tensor
    """
    glove = open(path, 'rb')

    # Forming the Glove Dictionary
    gloveDict = {}
    print("Creating the glove dictionary...")
    for l in glove:
        line = l.decode().split()
        word = line[0]
        vector = torch.Tensor(np.array(line[1:], dtype=np.float32))
        gloveDict[word] = vector
    print("glove dictionary created!")


    embeddings = torch.zeros((vocab.numWords, pretrainedembeddingSize))
    for index in range(vocab.numWords):
        try:
            embeddings[index] = gloveDict[vocab.index2word[index]]
        except KeyError: # If word not presetn in glove embedding dictionary
            embeddings[index] = torch.Tensor(np.random.normal(scale = 0.6, size = (pretrainedembeddingSize,)))

    return embeddings




