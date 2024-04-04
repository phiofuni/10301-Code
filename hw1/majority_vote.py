import sys

def get_maj(file_name):
    label_0 = 0
    label_1 = 0
    with open(file_name,'r') as file:
        for line in file:
            if line[-2]=='0':
                label_0+=1
            elif line[-2]=='1':
                label_1+=1
    file.close()
    return label_0,label_1

def get_train_statistic(file_name):
    label_0,label_1 = get_maj(file_name)
    total = label_0+label_1
    if label_0>label_1:
        maj = 0
        error_rate = label_1/total
    else:
        maj = 1
        error_rate = label_0/total
    return total,maj,error_rate

def get_test_statistic(maj,file_name):
    label_0,label_1 = get_maj(file_name)
    total = label_0+label_1
    if maj == 0:
        error_rate = label_1/total
    else:
        error_rate = label_0/total
    return total,error_rate

if __name__ == '__main__':
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics = sys.argv[5]
    
    train_total,train_maj,train_error_rate = get_train_statistic(train_in)
    test_total,test_error_rate = get_test_statistic(train_maj,test_in)

    metrics_string = f"error(train): {train_error_rate}\nerror(test): {test_error_rate}"
    with open(metrics,'w') as file:
        file.write(metrics_string)

    train_out_string = ""
    for i in range(train_total):
        if i==train_total-1:
            train_out_string = train_out_string+f"{train_maj}"
        else:
            train_out_string = train_out_string+f"{train_maj}\n"
    with open(train_out,'w') as file:
        file.write(train_out_string)
    
    test_out_string = ""
    for i in range(test_total):
        if i== test_total-1:
            test_out_string = test_out_string+f"{train_maj}"
        else:
            test_out_string = test_out_string+f"{train_maj}\n"
    with open(test_out,'w') as file:
        file.write(test_out_string)
    


