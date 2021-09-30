# CNN-Transformer



# introduction
CNN-Transformer is a hybrid deep learning method based on CNN and Transformer, used to predict orphan genes of moso bamboo.It is implemented by pytorch and all main scripts are written by Python on PC under a Microsoft Windows 10 operating system.

# Dependency
windows operating system

python 3.8.2

Pytorch

# Datasets.zip
"Phyllostachys_edulis_V1.pep" is the moso bamboo protein data downloaded on the Bamboo GDB official website.
"train.csv" is the training set
"valid.csv" is the validation set
"test.csv" is the test set
"maozhu_protein.csv"is a csv file of protein data of moso bamboo.
# CNN-Transformer.py
CNN-Transformer.py is the model implementation and training code and valid code of CNN-Transformer.he output are test results, including true positive (TP), false positive (FP), true negative (TN), false negative(FN), Balance accuracy (BA),  Geometric Mean(GM), matthews correlation coefficient (MCC), bookmaker informedness (BM) .

# test.py
test.py is the model implementation and testing code of CNN-Transformer.he output are test results, including true positive (TP), false positive (FP), true negative (TN), false negative(FN), Balance accuracy (BA),  Geometric Mean(GM), matthews correlation coefficient (MCC), bookmaker informedness (BM) .
