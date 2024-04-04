import sys
import math

def get_stat(file_name):
    label_0 = 0
    label_1 = 0
    error = 0
    with open(file_name,'r') as file:
        for line in file:
            if line[-2]=='0':
                label_0+=1
            elif line[-2]=='1':
                label_1+=1
    file.close()
    total = label_0+label_1
    prob_0 = label_0/total
    prob_1 = label_1/total
    if(label_0>label_1):
        error = prob_1
    else:
        error = prob_0

    entropy = -prob_1*math.log(prob_1,2)-prob_0*math.log(prob_0,2)
    
    
    return error,entropy


if __name__ == '__main__':
    file_in = sys.argv[1]
    file_out = sys.argv[2]

    error,entropy = get_stat(file_in)
    with open(file_out,'w') as file:
        file.write(f"entropy: {entropy}\nerror: {error}")
    file.close

