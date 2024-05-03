"""
Authors: del Vescovo Samuele, Lopopolo Antonio
"""
import matplotlib.pyplot as plt
import pandas


def load(path):
    """
    This function is useful to load the dataset.

    :param path: the path of dataset

    :return dataset: the dataframe which is the dataset
    """
    from pandas import read_csv
    dataset = read_csv(path)
    return dataset


def preElaborationData(dataset, list_attributes):
    """
    This function is useful to describe individuals attribute statistically
    and then understand for each attribute the distribution of values ​​with respect to the classes (via boxplot).
    The boxplot are useful to understanding which attributes seem most relevant to predict a certain class.
    
    :param dataset: the dataframe which is the dataset
    :param list_attributes: the list of name's attributes (list of strings)

    :return:
    """
    for a in list_attributes:
        print(dataset[a].describe(), "\n")

    import matplotlib.pyplot as plt
    for a in list_attributes:
        dataset.boxplot(column=a, by='Label')
        plt.show()


def removeColumns(dataset, list_attributes):
    """
    This function is useful to eliminate the attributes that has the same min and max 
    value because they are considered useless.

    :param dataset: the dataframe which is the dataset
    :param list_attributes: the list of name's attributes (list of strings)

    :return d: the dataset without useless independent attribute
    :return list_attributes: the independent attribute list is the same list of "list_attributes" without useless attribute
    """
    list_attributes_rem = list()
    for a in list_attributes:
        if dataset[a].max() == dataset[a].min():
            list_attributes_rem.append(a)
    d = dataset.drop(list_attributes_rem, axis=1)

    for a in list_attributes_rem:
        list_attributes.remove(a)

    return d, list_attributes


def preElaborationClass(dataset, label):
    """
    This function is useful to print the number of examples per label and an histogram
    to understand the probability distribution over the classes.

    :param dataset: the dataframe which is the dataset
    :param label: the string "Label"

    :return:
    """
    import matplotlib.pyplot as plt
    p = dataset[label].plot(x=label, title=label, kind="hist")
    plt.show()

    data = dataset.groupby(by=label).count()
    data = data.iloc[:, 0:1]
    print(data)


def topFeatureSelect(sorted_attr_dict, N):
    """
    This function is useful to select the first N attribute more relevant.

    :param sorted_attr_dict: the dictionary as a list of coupple <feature,rank>
    :param N: the number of feature to select
    
    :return list_attributes: list of first N attributes more relevant
    """
    list_attributes = list()
    for i in range (N):
        list_attributes.append(sorted_attr_dict[i][0])
    return list_attributes


def order_dataset_mutualInfoRank(dataset, list_attributes_w, label, N, seed):
    """
    This function is useful to sort the dataset via MI rank and select the first N features.

    :param dataset: the dataframe which is the dataset
    :param list_attributes_w: list of the features name without "Label"
    :param label: the string "Label"
    :param N: the number of more relevant features
    :param seed: the seed for pseudocasual random number generator

    :return selected_dataset : the dataset (dataframe) sorted via MI rank
    :return toplist: the first N features name (with "Label")
    """

    sorted_attr_dict = mutualInfoRank(dataset, list_attributes_w, label, seed)
    toplist = topFeatureSelect(sorted_attr_dict, N)
    toplist.append(label)
    selected_dataset = dataset.loc[:, toplist]
    return sorted_attr_dict, selected_dataset, toplist


def mutualInfoRank (data, independentList, label, seed):
    """
    This function is useful to create and sort one dictionary of coupple <feature, MI rank> in 
    descending order of MI rank.

    :param data: the dataframe which is the dataset
    :param independentList: list of the features name without "Label"
    :param label: the string "Label"
    :param seed: the seed for pseudocasual random number generator
    
    :return sorted_x: the dictionary sorted 
    """
    from sklearn.feature_selection import mutual_info_classif
    res = dict(zip(independentList, mutual_info_classif(data[independentList], data[label], discrete_features=False, random_state=seed)))
    sorted_x = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_x


def data_pca(dataset, list_attributes_w, label):
    """
    This function is useful to compute the PCA on dataset (along the independent variables) and to add the "Label"
    column to dataset. Every feature is named "pc_i" (i>=0 and i<=65).

    :param dataset: the dataframe which is the dataset
    :param list_attributes_w: list of the features name without "Label"
    :param label: the string "Label"

    :return pca_dataset: the dataset to which the PCA was applied
    :return model_pca: the PCA model
    :return pca_list_attributes: the features name's list (after PCA) without "Label"
    """
    X = dataset.loc[:, list_attributes_w]
    model_pca, pca_list_attributes = pca(X, list_attributes_w)
    pca_dataset = applyPCA(X, model_pca, pca_list_attributes)
    pca_dataset.insert(loc=len(list_attributes_w), column=label, value=dataset[label], allow_duplicates=True)
    return pca_dataset, model_pca, pca_list_attributes


def pca(data, list_attributes_w):
    """
    This function is useful to apply PCA and to rename every feature with "pc_i" (i>=0 and i<=65).

    :param data: the dataframe (dataset) to which to apply the PCA
    :param list_attributes_w: the feature name's list (without "Label")

    :return pca: the PCA model
    :return l: the new features name's list
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=len(list_attributes_w))
    pca.fit(data)
    l = list()
    for c in range (len(data.columns.values)):
        v = "pc_" + str(c+1)
        l.append(v)
    return pca, l


def applyPCA (dataset, model, pca_list_attributes):
    """
    This function is useful to apply PCA through model to dataset.

    :param dataset: the dataframe (dataset) to which to apply the PCA
    :param model: the PCA model
    :param pca_list_attributes: the feature name's list (without "Label") after PCA

    :return pca_dataframe: the dataset on which the PCA was applied
    """
    pca_dataset = model.transform(dataset)
    pca_dataframe = pandas.DataFrame(data=pca_dataset, columns=pca_list_attributes)
    return pca_dataframe


def selectedPCAData(dataset, N, label, list_attributes):
    """
    This function is useful to create one dataset with only N principal componencts (with "Label").

    :param dataset: the dataset (dataframe)
    :param N: the top N feature to select
    :param label: the string "Label" (name of the dependent attribute)
    :param list_attributes: the features name's list
   
    :return new_dataset: the new dataset with only N principal componencts (with "Label")
    """
    new_list = list()
    for x in range (0, N):
        new_list.append(list_attributes[x])
    new_dataset = dataset.loc[:, new_list]
    new_dataset.insert(loc=N, column=label, value=dataset[label], allow_duplicates=True)
    return new_dataset


def stratifiedKfold(X, y, fold, seed):
    """
    This function is useful to create a stratified k fold cross validation.
    
    :param X: the dataframe (dataset) along the independent variables (without the label)
    :param y: the dataframe (dataset) along the dependent variable
    :param fold: the fold number
    :param seed: the seed for pseudocasual random number generator

    :return ListXTrain: the list of "fold" position containing the training examples (along the independent variables) for each fold
    :return ListXTest: the list of "fold" position containing the test examples (along the independent variables) for each fold
    :return ListYTrain: the list of "fold" position containing the training examples (along the dependent variable) for each fold
    :return ListYTest: the list of "fold" position containing the test examples (along the dependent variable) for each fold
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    ListXTrain = list()
    ListXTest = list()
    ListYTrain = list()
    ListYTest = list()

    for train_index, test_index in skf.split(X, y):
        ListXTrain.append(X.iloc[train_index])
        ListXTest.append(X.iloc[test_index])
        ListYTrain.append(y.iloc[train_index])
        ListYTest.append(y.iloc[test_index])
    return ListXTrain, ListXTest, ListYTrain, ListYTest


def order_dataset_infogain(dataset, list_attributes_w, label, N):
    """
    This function is useful to sort the dataset through IG rank in descending order. The function returns a 
    dictionary of couples <feature, IG value> sorted in descending order.

    :param dataset: the dataframe (dataset)
    :param list_attributes_w: the features name's list (without label)
    :param label: the string "Label"
    :param N: the number of top features

    :return sorted_attr_dict: the dictionary of couples <feature, IG value> sorted in descending order
    :return selected_dataset: the dataframe along the first N features
    :return toplist: the top N features name's list (with label) ordered through IG
    """

    sorted_attr_dict = giRank(dataset, list_attributes_w, label)
    toplist = topFeatureSelect(sorted_attr_dict, N)
    toplist.append(label)
    selected_dataset = dataset.loc[:, toplist]
    return sorted_attr_dict, selected_dataset, toplist


def giRank(data, independentList, label):
    """
    This function is useful to create the dictionary of couples <feature, IG rank> sorted in descending order.

    :param data: the dataframe (dataset)
    :param independentList: the feature name's list (independent variables)
    :param label: the string "Label"

    :return sorted_x: the dictionary of couples <feature, IG rank> sorted in descending order
    """

    res = dict(zip(independentList, giClassif(data[independentList], data[label])))
    sorted_x = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_x


def giClassif(data, label):
    """
    This function is useful to create the IG rank list for independent variables (in each position there is the 
    IG rank for i-th feature).

    :param data: the dataframe (dataset) along the independent variables (without label)
    :param label: the dataframe (dataset) along the dependent variable
    
    :return info: the IG rank list for independent variables
    """
    cols = list(data.columns.values)
    info = []
    for c in cols:
        info.append(infogain(data[c], label))
    return info


def infogain(column, label):
    """
    This function is useful to calculate the IG rank for one feature (column) with rispect to label.
    
    :param column: the Series that represents the feature (column) to calculate the rank
    :param label: the Series that represents the label

    :return inf_gain: the IG rank for "column"
    """
    sum_prob = sum_entropy_classes(label)
    sum_prob_cond = w_sum_entropy_cond(column, label)
    inf_gain = sum_prob - sum_prob_cond
    return inf_gain


def sum_entropy_classes(label):
    """
    This function is useful to calculate the sum of entropies of classes.

    :param label: the Series that represent the label
    
    :return sum_entropy: the sum of entropies for each class (prior entropy on dataset)
    """
    sum_entropy = 0
    number_classes = label.nunique()
    probs = list()
    for i in range (number_classes):
        p = (label.tolist().count(i)) / (len(label.tolist()))
        probs.append(p)

    for prob in probs:
        sum_entropy = sum_entropy + entropy(prob, number_classes)
    return sum_entropy


def entropy(prob, n_classes):
    """
    This function calculate the entropy of the probability given in input.

    :param prob: the probability
    :param n_classes: the number of classes in dataset
    
    :return entr: the entropy of probability in input
    """
    from math import log
    entr = (-prob) * log(prob, n_classes)
    if entr == -0.0:
        return 0.0
    return entr


def w_sum_entropy_cond(column, label):
    """
    This function is useful to calculate the weighted mean of the single entropy sums 
    (for each value of the feature in exam).
    
    :param column: the Series that represents the feature (column) to calculate the rank
    :param label: the Series that represent the label
    
    :return weighted_mean: the value of weighted mean
    """

    #Create one dataframe with to columns: feature and label
    aux = pandas.concat([column, label], axis=1)

    #list_coloumn_names_aux è una lista di nomi delle colonne di aux
    #list_attribute is the list of the uniwue values of "column"
    list_coloumn_names_aux = aux.columns.values.tolist()
    number_classes = label.nunique()
    list_attribute = aux[list_coloumn_names_aux[0]].unique().tolist()

    #aux1 is a dataframe useful to calculate for each features values and label's values the examples that group
    #examples is the list of number of examples in each group 
    #indexes is the list of couples <features value, label's value> for each value
    aux1 = aux.groupby([list_coloumn_names_aux[0], list_coloumn_names_aux[1]]).size()
    examples = aux1.values.tolist()
    indexes = aux1.index.tolist()

    #Calculate the single prior probabilities
    aux1_list = list()
    probs = list()
    for i in range (len(examples)):
        probs.append(examples[i] / column.tolist().count(indexes[i][0]))

    #Create a list of triple <features value, label's value, prior probability>
    for i in range(len(probs)):
        temp = [indexes[i][0], indexes[i][1], probs[i]]
        aux1_list.append(list(temp))

    #Calculate for each probability the entropy
    for i in range (len(aux1_list)):
        aux1_list[i][2] = entropy(aux1_list[i][2], number_classes)

    #Create a list of couple <features value, sum of prior entropies
    list_sum = list()
    for att_value in list_attribute:
        sum_entropies = 0
        for i in range (len(aux1_list)):
            if aux1_list[i][0] == att_value:
                sum_entropies = sum_entropies + aux1_list[i][2]
        temp = [att_value, sum_entropies]
        list_sum.append(temp)

    #Calculate the sum of priori entropies weighted on the number of examples (of one specific class)
    weighted_sum = 0
    for i in range(len(list_sum)):
        weighted_sum = weighted_sum + (list_sum[i][1] * aux[list_coloumn_names_aux[0]].tolist().count(list_sum[i][0]))

    weighted_mean = weighted_sum / column.count()
    return weighted_mean


def decisionTreeLearner (X, y, c, min_samples_split):
    """
    This function is useful to learn the decision tree pattern based on dataset given in input.

    :param X: the dataframe (dataset) along the independent variables (without the label)
    :param y: the dataframe (dataset) along the dependent variable
    :param c: the criterion of tree (string)
    :param min_samples_split: the minimum number of examples to split the dataset
    
    :return T: the tree's pattern
    """
    from sklearn.tree import DecisionTreeClassifier
    T = DecisionTreeClassifier(criterion=c, splitter="best", random_state=0, min_samples_split=min_samples_split)
    """
    The "splitter" is best in order to choose the split that maximize the entropy
    The "random_state" is setted in order to shuffle the examples in the same way (for each run of software)
    """
    T.fit(X, y)
    return T


def showTree(T):
    """
    This function is useful to print some details of tree's pattern.
    
    :param T: the tree's pattern to print
    
    :return:
    """

    print("Numero di nodi",T.tree_.node_count)
    print("Numero di foglie", T.tree_.n_leaves)


def plotTree(T):
    """
    This function is useful to plot the tree's pattern .

    :param T: the tree's pattern to plot
    
    :return:
    """
    from sklearn import tree
    plt.figure(figsize=(20, 20))
    tree.plot_tree(T, fontsize=6, proportion=True)
    plt.show()


def decisionTreeF1(T, XTest, yTest):
    """
    This function is useful to calculate the f1-score (weighted) for the tree's pattern.
    
    :param T: the tree's pattern useful to calculate the f1-score (weighted)
    :param XTest: the dataframe (dataset) along the independent variables (without the label)
    :param yTest: the dataframe (dataset) along the dependent variable
    
    :return f1: the f1-score (weighted)
    """
    from sklearn.metrics import f1_score
    y_pred = T.predict(XTest)
    f1 = f1_score(yTest, y_pred, average='weighted')
    return f1


def decisionTree(T, XTest):
    """
    This function is useful to test the tree's pattern and to return the predictions
    
    :param T: the tree's pattern
    :param XTest: the dataframe (dataset) along the independent variables (without the label)
    
    :return y_pred: the predictions (array)
    """""
    y_pred = T.predict(XTest)
    return y_pred


def determineDecisionTreekFoldConfiguration(ListXTrain, ListYTrain, ListXTest, ListYTest, rank):
    """
    This function is useful to determine the best configuration of the tree's pattern. The configuration is:
    best criterion of splitting, best number of feature to train the model, best f1-score (the score is a mean of
    the 5 f1-score).

    :param ListXTrain: the list of "fold" position containing the training examples (along the independent variables) for each fold
    :param ListYTrain: the list of "fold" position containing the training examples (along the dependent variable) for each fold
    :param ListXTest: the list of "fold" position containing the test examples (along the independent variables) for each fold
    :param ListYTest: the list of "fold" position containing the test examples (along the dependent variable) for each fold
    :param rank: the ranking choosen (dictionary of couples <features name, features rank>) without label 
    
    :return bestCriterionMIR: best criterion to train the model (entropy or gini)
    :return bestMIR: best number of features
    :return bestEvalMIR: best f1-score corresponding to the best criterion
    """

    #Create the list of features names (from rank)
    list_features = list()
    for el in rank:
        list_features.append(el[0])

    #Create Creo delle liste di dataframe --> in pratica convero ogni lista della cv in liste di dataframe
    list_dataframe_XTrain = list()
    list_dataframe_YTrain = list()
    list_dataframe_XTest = list()
    list_dataframe_Ytest = list()

    #Convert ListXTrain, ListYTrain, ListXTest, ListYTest in list of dataframe  
    for dataset in ListXTrain:
        list_dataframe_XTrain.append(pandas.DataFrame(dataset))

    for column in ListYTrain:
        list_dataframe_YTrain.append(pandas.DataFrame(column))

    for dataset in ListXTest:
        list_dataframe_XTest.append(pandas.DataFrame(dataset))

    for column in ListYTest:
        list_dataframe_Ytest.append(pandas.DataFrame(column))

    #Create the lists useful to return the best configuration
    best_configuration_gini = list()
    best_configuration_entropy = list()

    criterions = list(['gini', 'entropy'])
    for criterion in criterions:
        #Creo una lista per memorizzare gli esiti delle cv (di posizioni 14) 
        #dove ogni posizione è una coppia [numero features, f1 media]
        #
        list_number_features_mean_f1 = list()
        for number_features in range(5, 70, 5):
            sum_f1 = 0
            for i in range(0, 5):
                #Perform the CV e sum the 5 f1-score calculated
                XTrain = list_dataframe_XTrain[i].loc[:, list_features[0:number_features]]
                yTrain = list_dataframe_YTrain[i]
                min_samples_split = 500
                T = decisionTreeLearner(XTrain, yTrain, criterion, min_samples_split)

                XTest = list_dataframe_XTest[i].loc[:, list_features[0:number_features]]
                yTest = list_dataframe_Ytest[i]
                f1 = decisionTreeF1(T, XTest, yTest)
                sum_f1 = sum_f1 + f1
            #Perform the mean of the 5 f1-score
            mean_f1 = sum_f1 / 5
            #Pushback of couple <actual features number, f1-score mean>
            list_number_features_mean_f1.append([number_features, mean_f1])

        #The last configuration with only one independent variable added
        sum_f1 = 0
        for i in range(0, 5):
            XTrain = list_dataframe_XTrain[i].loc[:, list_features]
            yTrain = list_dataframe_YTrain[i]
            min_samples_split = 500
            T = decisionTreeLearner(XTrain, yTrain, criterion, min_samples_split)

            XTest = list_dataframe_XTest[i].loc[:, list_features]
            yTest = list_dataframe_Ytest[i]
            f1 = decisionTreeF1(T, XTest, yTest)
            sum_f1 = sum_f1 + f1
        #Calculate the mean of the f1-score
        mean_f1 = sum_f1 / 5
        #Pushback of couple <actual features number, f1-score mean>
        list_number_features_mean_f1.append([len(list_features), mean_f1])
        #Choose the best configuration for gini anche entropy
        if criterion == 'gini':
            best_configuration_gini.append(max(list_number_features_mean_f1, key=lambda x: x[1]))
        else:
            best_configuration_entropy.append(max(list_number_features_mean_f1, key=lambda x: x[1]))

    #Choose the best configuration and the best criterion of splitting considering the maximun f1-score mean

    bestCriterion = ''
    bestN = 0
    bestEval = 0

    if best_configuration_gini[0][1] > best_configuration_entropy[0][1]:
        bestCriterion = 'gini'
        bestN = best_configuration_gini[0][0]
        bestEval = best_configuration_gini[0][1]
    elif best_configuration_gini[0][1] < best_configuration_entropy[0][1]:
        bestCriterion = 'entropy'
        bestN = best_configuration_entropy[0][0]
        bestEval = best_configuration_entropy[0][1]
    else:
        if best_configuration_gini[0][0] < best_configuration_entropy[0][0]:
            bestCriterion = 'gini'
            bestN = best_configuration_gini[0][0]
            bestEval = best_configuration_gini[0][1]
        elif best_configuration_entropy[0][0] < best_configuration_gini[0][0]:
            bestCriterion = 'entropy'
            bestN = best_configuration_entropy[0][0]
            bestEval = best_configuration_entropy[0][1]
        else:
            import random
            c = random.uniform(0, 1)
            if c >= 0.5:
                bestCriterion = 'gini'
                bestN = best_configuration_gini[0][0]
                bestEval = best_configuration_gini[0][1]
            else:
                bestCriterion = 'entropy'
                bestN = best_configuration_entropy[0][0]
                bestEval = best_configuration_entropy[0][1]

    return bestCriterion, bestN, bestEval


def classReport(yTest, ypred, labels):
    """
    This function is useful to obtain the classification report. 

    :param yTest: the ground through (array)
    :param ypred: the prection of pattern (array)
    :param labels: the list of label's unique value
    
    :return cr: the classification report (string)
    """
    from sklearn.metrics import classification_report
    cr = classification_report(yTest, ypred, labels= labels)
    return cr


def confMatrix(yTest, ypred, labels):
    """
    The function is useful to compute the confusion matrix.

    :param yTest: the ground through (array)
    :param ypred: the prection of pattern (array)
    :param labels: the list of label's unique value
    
    :return cm: the confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yTest, ypred, labels=labels)
    print(cm)
    return cm


def plotconfMatrix(confusionMatrix, labels):
    """
    This function is useful to plot the confusion matrix.

    :param confusionMatrix: the confusion matrix (array)
    :param labels: the list of label's unique value
    
    :return:
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels = labels)
    disp.plot()
    plt.show()

def trainTestPattern(rank, bestR, type_rank, train_dataset, label, bestCriterionR, test_dataset):
    """
    In questa funzione vado ad ad apprendere un pattern sul training set e lo vado a testare sul train set (specificati in input),
    calcolo il classification report per il testing sia sui valori della variabile target del training set che si quelli della variabile target
    del testing set.
    Mostro la matrice di confusione ottenuta dal testing del pattern sui valori della variabile target sono del testing set

    :param rank: dictionary of couple <features name, feature rank>
    :param bestR: best number of feature (from CV)
    :param type_rank: the type of rank (string) i.e. "MI", "IG" or "PCA". This indicate the rank on which the pattern is learned
    :param train_dataset: the training dataset (with label)
    :param label: the string "Label"
    :param bestCriterionR: the best criterion of splitting ('gini' o 'entropy')
    :param test_dataset: the testing dataset (with label)
    
    :return:
    """
    list_best_attributes = topFeatureSelect(rank, bestR)
    labelsTrain = list(set(train_dataset[label].tolist()))

    #Train Phase
    XTrain = train_dataset.loc[:, list_best_attributes]
    yTrain = train_dataset[label]
    min_samples_split = 500
    c = bestCriterionR

    #Learning the pattern
    T = decisionTreeLearner(XTrain, yTrain, c, min_samples_split)

    # Show some information about the pattern learned like: criterion, rank, features number,
    # node's number, leaf's number
    print("Tree: Criterion= {criterion}, rank= {rank}, first {numberf} features".format(
        criterion=bestCriterionR, rank=type_rank, numberf=bestR))
    showTree(T)

    #Compute the predictions on the training set
    ypredTrain = decisionTree(T, XTrain)

    #Plot the classification report of the pattern learned on the training set and tested on the training set
    print("Classification Report (Train): Criterion= {criterion}, rank= {rank}, first {numberf} features".format(
        criterion=bestCriterionR, rank=type_rank, numberf=bestR))
    print(classReport(yTrain, ypredTrain, labels=labelsTrain))

    #Testing Phase
    labelsTest = list(set(test_dataset[label].tolist()))
    XTest = test_dataset.loc[:, list_best_attributes]
    yTest = test_dataset[label].tolist()

    #Compute the predictions on the testing dataset
    ypredTest = decisionTree(T, XTest)

    #Plot the classification report of the pattern learned on the training set and tested on the testing set
    print("Classification Report (Test): Criterion= {criterion}, rank= {rank}, first {numberf} features".format(
        criterion=bestCriterionR, rank=type_rank, numberf=bestR))
    print(classReport(yTest, ypredTest, labels=labelsTest))
    print("Confusion Matrix: Criterion= {criterion}, rank= {rank}, first {numberf} features".format(
        criterion=bestCriterionR, rank=type_rank, numberf=bestR))
    cm = confMatrix(yTest, ypredTest, labels=labelsTest)
    plotconfMatrix(cm, labelsTest)
    print("\n")

