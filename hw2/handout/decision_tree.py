import argparse
import numpy as np
import math

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self,i=-1,a='leaf',array=None):
        self.left = None
        self.right = None
        self.attr = a
        self.index = i
        self.vote = None
        self.array = array
        self.acc = [i]
    
    def print_tree(self,level=1):
        s = ""
        if level == 1:
            num1 = self.array[:,-1].sum()
            num0 = self.array[:,-1].size-num1
            s = f"[{num0} 0/{num1} 1]\n"
        if self.left != None:
            left = self.array[:,-1][self.array[:,self.index] == 0]
            left_1 = left.sum()
            left_0 = left.size - left_1
            pre = "| "*level
            s += pre[:-1]+f" {self.attr} = 0: [{left_0} 0/{left_1} 1]\n"
            s += self.left.print_tree(level+1)
        if self.right != None:
            right = self.array[:,-1][self.array[:,self.index] == 1]
            right_1 = right.sum()
            right_0 = right.size - right_1
            pre = "| "*level
            s += pre[:-1]+f" {self.attr} = 1: [{right_0} 0/{right_1} 1]\n"
            s += self.right.print_tree(level+1)
        return s

def get_entropy(l):
    e = 0
    size = l.size
    if size == 0:
        return 0
    total_sum = l.sum()
    prob_1 = total_sum/size
    prob_0 = (size-total_sum)/size
    if prob_1 == 0 or prob_1 == 1:
        e = 0
    else:    
        e = -prob_1*math.log(prob_1,2)-prob_0*math.log(prob_0,2)
    return e

def get_given_entropy(l, c):
    e = 0
    size = l.size
    if size == 0:
        return 0

    c_given_l1 = c[l == 1]
    c_given_l0 = c[l == 0]
    
    cl1_size = c_given_l1.size
    cl0_size = c_given_l0.size
    
    if cl1_size == 0:
        cl1_e = 0
    else:
        cl1_sum = c_given_l1.sum()
        cl1_prob1 = cl1_sum/cl1_size
        cl1_prob0 = (cl1_size-cl1_sum)/cl1_size
        if cl1_prob1==1 or cl1_prob1==0:
            cl1_e = 0
        else:
            cl1_e = -cl1_prob1*math.log(cl1_prob1,2)-cl1_prob0*math.log(cl1_prob0,2)
        
    if cl0_size == 0:
        cl0_e = 0
    else:
        cl0_sum = c_given_l0.sum()
        cl0_prob1 = cl0_sum/cl0_size
        cl0_prob0 = (cl0_size-cl0_sum)/cl0_size
        if cl0_prob1 == 1 or cl0_prob1 == 0:
            cl0_e = 0
        else:
            cl0_e = -cl0_prob1*math.log(cl0_prob1,2)-cl0_prob0*math.log(cl0_prob0,2)
    
    e = (cl1_size/size)*cl1_e+(cl0_size/size)*cl0_e
    return e

def get_mi(given,y):
    return get_entropy(y)-get_given_entropy(given,y)

def get_maj(l):
    if (l.size-l.sum())>l.sum():
        return 0
    return 1

def make_maj_vote(var):
    node = Node()
    node.vote = var
    return node

def build_tree(node,attr_l,depth):
    if depth == 0:
        left_node = Node()
        l_y = node.array[:,-1][node.array[:,node.index]==0]
        left_node.vote = get_maj(l_y)
        node.left = left_node
        
        right_node = Node()
        l_r = node.array[:,-1][node.array[:,node.index]==1]
        right_node.vote = get_maj(l_r)
        node.right = right_node
        return ;
    
    left_array = node.array[node.array[:,node.index] == 0]
    right_array = node.array[node.array[:,node.index] == 1]
    left_array_end = left_array[:,-1]
    right_array_end = right_array[:,-1]

    left_h = get_entropy(left_array_end)
    right_h = get_entropy(right_array_end)

    if left_array_end.sum() == 0:
        l_var = 0
    else:
        l_var = 1
    
    if right_array_end.sum() == 0:
        r_var = 0
    else:
        r_var = 1

    if left_h == 0 and right_h == 0:
        
        node.left = make_maj_vote(l_var)
        node.right = make_maj_vote(r_var)
        return ;
    
    l_index = 0
    l_max = 0
    r_max = 0
    r_index = 0
    for i in range(len(attr_l)-1):
        if i!= node.index and i not in node.acc:

            l_mi = get_mi(left_array[:,i],left_array[:,-1])
            r_mi = get_mi(right_array[:,i],right_array[:,-1])

            if l_mi>l_max:
                l_max = l_mi
                l_index = i
            if r_mi>r_max:
                r_max = r_mi
                r_index = i

    if left_h == 0:
        node.left = make_maj_vote(l_var)
        r_node = Node(r_index,attr_l[r_index])
        r_node.array = right_array
        r_node.acc += node.acc
        node.right = r_node
        build_tree(r_node,attr_l,depth-1)

    elif right_h == 0:
        node.right = make_maj_vote(r_var)
        l_node = Node(l_index,attr_l[l_index])
        l_node.array = left_array
        l_node.acc += node.acc
        node.left = l_node
        build_tree(l_node,attr_l,depth-1)

    else:
        r_node = Node(r_index,attr_l[r_index])
        r_node.array = right_array
        r_node.acc+=node.acc
        node.right = r_node

        l_node = Node(l_index,attr_l[l_index])
        l_node.array = left_array
        l_node.acc+=node.acc
        node.left = l_node

        build_tree(l_node,attr_l,depth-1)
        build_tree(r_node,attr_l,depth-1)
        


def process(file_name,max_depth):
    
    array = []
    with open(file_name,'r') as file:
        for line in file:
            attr_list = line.strip().split()
            # print(attr_list)
            break
        
        for line in file:
            each = line.strip().split()
            int_line = [int(x) for x in each]
            array.append(int_line)

    array = np.array(array)

    if max_depth == 0:
        tree = make_maj_vote(get_maj(array[:,-1]))
        tree.array = array
        return tree

    max = 0
    index = 0
    for i in range(len(attr_list)-1):
        inf = get_mi(array[:,i],array[:,-1])
        if inf>max:
            max = inf
            index = i

    root = Node(index,attr_list[index],array)
    build_tree(root,attr_list,max_depth-1)
    # root.print_tree()
    return root

def predict(tree,line):
    cur = tree
    while(cur.left != None or cur.right != None):
        if line[cur.index] == 0:
            cur = cur.left
        else:
            cur = cur.right
    return cur.vote


def test(file_name,tree):
    array = []
    with open(file_name,'r') as file:
        for line in file:
            attr_list = line.strip().split()
            break
        
        for line in file:
            each = line.strip().split()
            int_line = [int(x) for x in each]
            array.append(int_line)

    array = np.array(array)
    predictions = []
    for line in array:
        predictions.append(predict(tree,line))
    dis = 0
    for i in range(len(predictions)):
        if predictions[i] != array[i,-1]:
            dis+=1
    error = dis/len(predictions)
    return predictions,error


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str,
                        help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()
    
    #Here's an example of how to use argparse
    train_input = args.train_input
    test_input = args.test_input
    train_out = args.train_out
    test_out = args.test_out
    metrics_out = args.metrics_out
    max_depth = args.max_depth
    print_out = args.print_out

    tree = process(train_input,max_depth)
    test_pred,test_error = test(test_input,tree)
    train_pred,train_error = test(train_input,tree)

    test_output = ""
    for x in test_pred:
        test_output += f"{x}\n"
    test_output = test_output[:-1]
    with open(test_out,'w') as file:
        file.write(test_output)
    
    train_output = ""
    for x in train_pred:
        train_output += f"{x}\n"
    train_output = train_output[:-1]
    with open(train_out,'w') as file:
        file.write(train_output)
    
    metric_s = f"error(train): {train_error}\nerror(test): {test_error}"
    with open(metrics_out,'w') as file:
        file.write(metric_s)
    
    with open(print_out,'w') as file:
        out = tree.print_tree()
        file.write(out[:-1])
    

    #Here is a recommended way to print the tree to a file
    # with open(print_out, "w") as file:
    #     print_tree(dTree, file)
