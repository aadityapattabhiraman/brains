#!/home/akugyo/Programs/Python/torch/bin/python

with open("../../Dataset/baby_names.txt") as f:
    data = f.readlines()


list = []

for i in data:
    i = i[:-2]
    content = i.split(",")
    if len(content) == 2:
        list.append(content[1] + "\n")
    else:
        for i in range(1, len(content)):
            list.append(content[i] + "\n")

string = ""
list = []

for i in data:
    list.append(i.split(",")[1][:-2])


with open("../../Dataset/data/boys.txt") as f:
    data_1 = f.readlines()

for i in data_1:
    if i[:-1] not in list:
        list.append(i[:-1] + "\n")
        list.append(i[:-1])

with open("../../Dataset/data/girls.txt") as f:
    data_2 = f.readlines()

for i in data_2:
    if i[:-1] not in list:
        list.append(i[:-1] + "\n")

with open("../../Dataset/names.txt", "w") as f:
    f.writelines(list)
        list.append(i[:-1])

for i in list:
    string += i + "\n"


with open("../../Dataset/names.txt", "w") as f:
    f.write(string)
