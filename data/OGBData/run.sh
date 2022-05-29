#Edited by Li Ling on May 28, 2022
#!/bin/sh
#dataName is data's name,such as arxiv,products...
dataName='arxiv'

#Data format conversion
python transOGBData_To_NeutronStarData.py ${dataName}

./convert2binary ${dataName}/Data/edge_data.txt $dataName/Data/${dataName}.edge.self.bin
