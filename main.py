"""
Authors: del Vescovo Samuele, Lopopolo Antonio
"""
import numpy.random
from functions import *

if __name__ == '__main__':
    #1 STEP
    train_path = "./dataset/trainDdosLabelNumeric.csv"
    test_path = "./dataset/testDdosLabelNumeric.csv"

    # First n features
    N = 10
    label = "Label"
    seed = 42
    fold = 5
    numpy.random.seed(seed)
    train_dataset = load(train_path)
    test_dataset = load(test_path)

    print("Report Dataset (Initial Version)")
    print("Number of examples: {exa} \n Number of attributes: {natt} \n Attributes: {att}".format(
        exa= len(train_dataset.values), natt= len(train_dataset.columns), att= train_dataset.columns.values))

    #2 STEP
    list_attributes = list(train_dataset.columns.values)
    #Commented for brevity. Uncomment to show boxplots.
    #preElaborationData(train_dataset, list_attributes)

    #3 STEP
    train_dataset_w, list_attributes = removeColumns(train_dataset, list_attributes)

    #4 STEP
    preElaborationClass(train_dataset_w, label)

    #5 STEP
    list_attributes_w = list_attributes.copy()
    list_attributes_w.remove(label)

    """
    The rank MI should be the same on dataset and the dataset scaled ... but they aren't the same. 
    """
    #Sort the training dataset via mutual info rank of the independent variables
    dataset_mutual_train = train_dataset_w.copy(deep=True)
    rankMI, selected_dataset_mutual_train, toplist_mutual = order_dataset_mutualInfoRank(dataset_mutual_train, list_attributes_w, label, N, seed)
    print("RANK MI", rankMI)

    #Sorting the training dataset via mutual info rank of the independent variables (after min max scaling)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    dataset_scaled = train_dataset_w.copy(deep=True)
    dataset_scaled[list_attributes_w] = scaler.fit_transform(dataset_scaled[list_attributes_w])
    rankMI_scaled, selected_dataset_scaled, toplist_scaled = order_dataset_mutualInfoRank(dataset_scaled, list_attributes_w, label, N, seed)

    print("Toplist", toplist_mutual)
    print("Toplist_scaled", toplist_scaled)
    print("Rank MI", rankMI)
    print("Rank MI Scaled", rankMI_scaled)
    print("RankMI is equal to Rank MI Scaled: ", rankMI==rankMI_scaled)

    """
    By setting the seed, the feature lists are the same
    (the scaled dataset and the unscaled dataset)
    """

    #6 STEP
    #Applying pca to dataset (as Sorting criterion)
    dataset_pca_train = train_dataset_w.copy(deep=True)
    dataset_pca_train, model_pca, list_attributes_pca = data_pca(dataset_pca_train, list_attributes_w, label)
    dataset_pca_selected = selectedPCAData(dataset_pca_train, N, label, list_attributes_pca)


    #7 STEP
    #Sorting the training dataset via info gain rank of the independent variables
    dataset_infogain_train = train_dataset_w.copy(deep=True)
    rankIG, selected_dataset_infogain, toplist_infogain = order_dataset_infogain(dataset_infogain_train, list_attributes_w, label, N)
    print("RANK IG:", rankIG)

    #8 STEP
    #Perform Cross Validation on dataset ordered via mutial info rank
    X = train_dataset_w.loc[:, list_attributes_w]
    y = train_dataset_w[label]
    ListXTrain, ListXTest, ListYTrain, ListYTest = stratifiedKfold(X, y, fold, seed)


    bestCriterionMIR, bestMIR, bestEvalMIR = determineDecisionTreekFoldConfiguration(ListXTrain, ListYTrain, ListXTest, ListYTest, rankMI)
    print("Best MI: Criterion = {criterion}, Number of features = {nf}, Best F1 (mean) {bestf1m}".format(
        criterion=bestCriterionMIR, nf=bestMIR, bestf1m=bestEvalMIR))

    #Perform Cross Validation on dataset ordered via info gain rank
    bestCriterionIGR, bestIGR, bestEvalIGR = determineDecisionTreekFoldConfiguration(ListXTrain, ListYTrain, ListXTest, ListYTest, rankIG)
    print("Best IG: Criterion = {criterion}, Number of features = {nf}, Best F1 (mean) {bestf1m}".format(
        criterion=bestCriterionIGR, nf=bestIGR, bestf1m=bestEvalIGR))


    #Perform Cross Validation on dataset ordered via PCA rank
    XPCA = dataset_pca_train.loc[:, list_attributes_pca]
    yPCA = dataset_pca_train[label]

    ListXPCATrain, ListXPCATest, ListYPCATrain, ListYPCATest = stratifiedKfold(XPCA, yPCA, fold, seed)

    l = list()
    for c in list_attributes_pca:
        l.append(1.0)
    res = dict(zip(list_attributes_pca, l))
    rankPCA = sorted(res.items(), key=lambda kv: kv[1], reverse=True)

    bestCriterionPCAR, bestPCAR, bestEvalPCAR = determineDecisionTreekFoldConfiguration(ListXPCATrain, ListYPCATrain, ListXPCATest, ListYPCATest, rankPCA)
    print("Best PCA: Criterion = {criterion}, Number of features = {nf}, Best F1 (mean) {bestf1m}".format(
        criterion=bestCriterionPCAR, nf=bestPCAR, bestf1m=bestEvalPCAR), "\n")



    #9 STEP

    #Train and Test on dataset ordered via MI rank
    trainTestPattern(rankMI, bestMIR, 'MI', train_dataset_w, label, bestCriterionMIR, test_dataset)

    #Train and Test on dataset ordered via IG rank
    trainTestPattern(rankIG, bestIGR, 'IG', train_dataset_w, label, bestCriterionIGR, test_dataset)

    #Train and Test on dataset ordered via PCA rank
    # The PCA model is the model derived from PCA on training set
    dataset_pca_test = applyPCA(test_dataset.loc[:, list_attributes_w], model_pca, list_attributes_pca)
    dataset_pca_test.insert(loc=len(list_attributes_w), column=label, value=test_dataset[label], allow_duplicates=True)

    trainTestPattern(rankPCA, bestPCAR, 'PCA', dataset_pca_train, label, bestCriterionPCAR, dataset_pca_test)


