# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/31 0031 11:43

import itertools


#   This function generates the first candidate set using the dataset
def generate_c1(data_set):
    product_dict = {}
    return_set = []
    for data in data_set:
        for product in data:
            if product not in product_dict:
                product_dict[product] = 1
            else:
                product_dict[product] = product_dict[product] + 1
    for key in product_dict:
        temp_array = []
        temp_array.append(key)
        return_set.append(temp_array)
        return_set.append(product_dict[key])
        temp_array = []
    return return_set


#   This function creates Frequent item sets by taking candidate sets as input
#   At the end, this function calls generateCandidatSets by feeding the output of the
#   current function as the input of the other function
def generate_frequent_item_set(candidate_list, no_of_transactions, minimum_support, data_set, father_frequent_array):
    frequentItemsArray = []
    for i in range(len(candidate_list)):
        if i % 2 != 0:
            support = (candidate_list[i] * 1.0 / no_of_transactions) * 100
            if support >= minimum_support:
                frequentItemsArray.append(candidate_list[i - 1])
                frequentItemsArray.append(candidate_list[i])
            else:
                eleminatedItemsArray.append(candidate_list[i - 1])

    for k in frequentItemsArray:
        father_frequent_array.append(k)

    if len(frequentItemsArray) == 2 or len(frequentItemsArray) == 0:
        # print("This will be returned")
        returnArray = father_frequent_array
        return returnArray

    else:
        generate_candidate_sets(data_set, eleminatedItemsArray, frequentItemsArray, no_of_transactions, minimum_support)


#   This function creates Candidate sets by taking frequent sets as the input
#   At the end, this function calls generateFrequentItemSets by feeding the output of the
#   crrent function as the input of the other function
def generate_candidate_sets(data_set, eleminatedItemsArray, frequent_items_array, no_of_transactions, minimum_support):
    only_elements = []
    array_after_combinations = []
    candidate_set_array = []
    for i_ in range(len(frequent_items_array)):
        if i_ % 2 == 0:
            only_elements.append(frequent_items_array[i])
    for item in only_elements:
        temp_combination_array = []
        k = only_elements.index(item)
        for i in range(k + 1, len(only_elements)):
            for j in item:
                if j not in temp_combination_array:
                    temp_combination_array.append(j)
            for m in only_elements[i]:
                if m not in temp_combination_array:
                    temp_combination_array.append(m)
            array_after_combinations.append(temp_combination_array)
            temp_combination_array = []
    sorted_combination_array = []
    unique_combination_array = []
    for i in array_after_combinations:
        sorted_combination_array.append(sorted(i))
    for i in sorted_combination_array:
        if i not in unique_combination_array:
            unique_combination_array.append(i)
    array_after_combinations = unique_combination_array
    for item in array_after_combinations:
        count = 0
        for transaction in data_set:
            if set(item).issubset(set(transaction)):
                count = count + 1
        if count != 0:
            candidate_set_array.append(item)
            candidate_set_array.append(count)
    generate_frequent_item_set(candidate_set_array, no_of_transactions, minimum_support, data_set, fatherFrequentArray)


#   This function takes all the frequent sets as the input and generates Association Rules
def generate_association_rule(freq_set):
    association_rule = []
    for item in freq_set:
        if isinstance(item, list):
            if len(item) != 0:
                length = len(item) - 1
                while length > 0:
                    combinations = list(itertools.combinations(item, length))
                    temp = []
                    lhs = []
                    for RHS in combinations:
                        lhs = set(item) - set(RHS)
                        temp.append(list(lhs))
                        temp.append(list(RHS))
                        # print(temp)
                        association_rule.append(temp)
                        temp = []
                    length = length - 1
    return association_rule


#   This function creates the final output of the algorithm by taking Association Rules as the input
def apriori_output(rules, data_set, minimum_support, minimum_confidence):
    return_apriori_output = []
    for rule in rules:
        support_of_x = 0
        support_of_xin_percentage = 0
        support_of_x_and_y = 0
        support_of_x_and_y_in_percentage = 0
        for transaction in data_set:
            if set(rule[0]).issubset(set(transaction)):
                support_of_x = support_of_x + 1
            if set(rule[0] + rule[1]).issubset(set(transaction)):
                support_of_x_and_y = support_of_x_and_y + 1
        support_of_xin_percentage = (support_of_x * 1.0 / noOfTransactions) * 100
        support_of_x_and_y_in_percentage = (support_of_x_and_y * 1.0 / noOfTransactions) * 100
        confidence = (support_of_x_and_y_in_percentage / support_of_xin_percentage) * 100
        if confidence >= minimum_confidence:
            support_of_x_append_string = "Support Of X: " + str(round(support_of_xin_percentage, 2))
            support_of_x_and_y_append_string = "Support of X & Y: " + str(round(support_of_x_and_y_in_percentage))
            confidence_append_string = "Confidence: " + str(round(confidence))

            return_apriori_output.append(support_of_x_append_string)
            return_apriori_output.append(support_of_x_and_y_append_string)
            return_apriori_output.append(confidence_append_string)
            return_apriori_output.append(rule)

    return return_apriori_output


#   These few statements are taking input from the user
#       Such as:
#           Select a database to mine the data
#           Minimum Support
#           Mnimum Confidence
print("Select from the following dataset:")
print("1. Auto Mobile")
print("2. Candies")
print("3. Computer Accesories")
print("4. Food")
print("5. Mobile Accesories")
print("\n")
fileNameInput = input("Enter number (1,2,3,4,5): ")
minimumSupport = input('Enter minimum Support: ')
minimumConfidence = input('Enter minimum Confidence: ')
print("\n")
fileName = ""

if fileNameInput == '1':
    fileName = "automobile.txt"
if fileNameInput == '2':
    fileName = "candies.txt"
if fileNameInput == '3':
    fileName = "computerStuff.txt"
if fileNameInput == '4':
    fileName = "food.txt"
if fileNameInput == '5':
    fileName = "mobileStuff.txt"

minimumSupport = int(minimumSupport)
minimumConfidence = int(minimumConfidence)

nonFrequentSets = []
allFrequentItemSets = []
tempFrequentItemSets = []
dataSet = []
eleminatedItemsArray = []
noOfTransactions = 0
fatherFrequentArray = []
something = 0

#   Reading the data file line by line
with open(fileName, 'r') as fp:
    lines = fp.readlines()

for line in lines:
    line = line.rstrip()
    dataSet.append(line.split(","))

noOfTransactions = len(dataSet)

firstCandidateSet = generate_c1(dataSet)

frequentItemSet = generate_frequent_item_set(firstCandidateSet, noOfTransactions, minimumSupport, dataSet,
                                             fatherFrequentArray)

associationRules = generate_association_rule(fatherFrequentArray)

AprioriOutput = apriori_output(associationRules, dataSet, minimumSupport, minimumConfidence)

counter = 1
if len(AprioriOutput) == 0:
    print("There are no association rules for this support and confidence.")
else:
    for i in AprioriOutput:
        if counter == 4:
            print(str(i[0]) + "------>" + str(i[1]))
            counter = 0
        else:
            print(i, end='  ')
        counter = counter + 1
