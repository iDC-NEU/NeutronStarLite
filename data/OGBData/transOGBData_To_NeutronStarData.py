import csv
import sys

import pandas as pd
import shutil
import os
import sys

if __name__ == '__main__':
    arg1 = sys.argv[1]
    dataName = arg1

    if not os.path.exists(dataName + '/tempData'):  # Whether the folder exists
        os.makedirs(dataName + '/tempData')

    #Copy a new edge .csv file to prevent subsequent operations from contaminating the original file
    shutil.copy(dataName + '/raw/edge.csv/edge.csv',dataName + '/tempData/edge_data_L.csv')

    #Added loop edge L
    with open(dataName + '/raw/num-node-list.csv/num-node-list.csv', "r") as my_input_file:
       for x in csv.reader(my_input_file):
            n = int(x[0])
    alist = list(range(n))

    with open(dataName + '/tempData/edge_data_L.csv', 'a') as my_output_file:
        for a in alist:
            my_output_file.write("".join(str(a)))
            my_output_file.write(",")
            my_output_file.write("".join(str(a))+"\n")
        my_output_file.close()

    #sort S
    df = pd.read_csv(dataName + '/tempData/edge_data_L.csv',header=None,names=['a','b'])
    data = df.sort_values(by='a')
    data.to_csv(dataName + '/tempData/edge_data_LS.csv', index=False, header=None)

    #Add a reverse edge R
    with open(dataName + '/tempData/edge_data_LSR.csv', "w") as my_output_file:
        with open(dataName + '/tempData/edge_data_LS.csv', "r") as my_input_file:
            for row in csv.reader(my_input_file):
                my_output_file.write("".join(row[1]) + "," + "".join(row[0]) + "\n")
                my_output_file.write("".join(row[0]) + "," + "".join(row[1]) + "\n")
        my_output_file.close()

    #Remove duplicate edges S
    df = pd.read_csv(dataName + '/tempData/edge_data_LSR.csv',header=None,names=['a','b'])
    data1 = df.drop_duplicates()
    data1.to_csv(dataName + '/tempData/edge_data_LSRD.csv', index=False, header=None)

    if not os.path.exists(dataName + '/Data'):  # hether the folder exists
        os.makedirs(dataName + '/Data')

    #csv to txt
    with open(dataName + '/Data/edge_data.txt', "w") as my_output_file:
        with open(dataName + '/tempData/edge_data_LSRD.csv', "r") as my_input_file:
            [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()

    #label trans
    n = 0
    with open(dataName + '/Data/node_label.txt', "w") as my_output_file:
        with open(dataName + '/raw/node-label.csv/node-label.csv', "r") as my_input_file:
            for row in csv.reader(my_input_file):
                my_output_file.write(str(n) + " " + " ".join(row)+'\n')
                n += 1
        my_output_file.close()

    #feat trans
    n = 0
    with open(dataName + '/Data/node_feat.txt', "w") as my_output_file:
        with open(dataName + '/raw/node-feat.csv/node-feat.csv', "r") as my_input_file:
            for row in csv.reader(my_input_file):
                my_output_file.write(str(n) + " " + " ".join(row)+'\n')
                n += 1
        my_output_file.close()

    #trian、eval、test add instructions
    if dataName == "products":
        splitname = "sales_ranking"
    elif dataName == "proteins" or dataName == "proteinfunc":
        splitname = "species"
    else:
        splitname = "time"
    with open(dataName + '/split/' + splitname + '/train.csv/train.csv', 'r') as csvFile:
      rows = csv.reader(csvFile)
      with open((dataName + '/tempData/train.csv'), 'w',newline='') as f:
        writer = csv.writer(f)
        for row in rows:
          row.append('train')
          writer.writerow(row)

    with open(dataName + '/split/' + splitname + '/valid.csv/valid.csv', 'r') as csvFile:
      rows = csv.reader(csvFile)
      with open((dataName + '/tempData/eval.csv'), 'w',newline='') as f:
        writer = csv.writer(f)
        for row in rows:
          row.append('eval')
          writer.writerow(row)

    with open(dataName + '/split/' + splitname + '/test.csv/test.csv', 'r') as csvFile:
      rows = csv.reader(csvFile)
      with open((dataName + '/tempData/test.csv'), 'w',newline='') as f:
        writer = csv.writer(f)
        for row in rows:
          row.append('test')
          writer.writerow(row)


    #merge train、eval、test
    filenames = [dataName + '/tempData/train.csv', dataName + '/tempData/eval.csv', dataName + '/tempData/test.csv']
    with open(dataName + '/tempData/mask_temp.csv', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


    #mask sort
    df = pd.read_csv(dataName + '/tempData/mask_temp.csv',header=None,names=['a','b'])
    data = df.sort_values(by='a')
    print(data)
    data.to_csv(dataName + '/tempData/mask_sort.csv', index=False, header=None)

    #mask trans2,csv to txt
    with open(dataName + '/Data/mask.txt', "w") as my_output_file:
        with open(dataName + '/tempData/mask_sort.csv', "r") as my_input_file:
            [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()


    #rename
    maskname = dataName + ".mask"
    os.rename(dataName + '/Data/mask.txt', dataName + '/Data/' + maskname)

    featname = dataName + ".featuretable"
    os.rename(dataName + '/Data/node_feat.txt', dataName + '/Data/' + featname)

    labelname = dataName + ".labeltable"
    os.rename(dataName + '/Data/node_label.txt', dataName + '/Data/' + labelname)
