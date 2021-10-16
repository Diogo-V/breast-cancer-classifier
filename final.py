# Import definition
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier

# Constants definition
GROUP_NUMBER = 16  # Our group number
NEIGHBOURS = [3, 5, 7]  # Number of neighbours to be used

# ----------------------------------------- PREPROCESSING AND DATA FILTERING ----------------------------------------- #

# Loading dataset into working desk
data = arff.loadarff('breast.w.arff')
df = pd.DataFrame(data[0])

# Removes NaN values from dataset by deleting rows
df.dropna(axis=0, how="any", inplace=True)

# Gets X (data matrix) and y (target values column matrix)
X = df.drop("Class", axis=1).to_numpy()
y = df["Class"].to_numpy()

# Performs some preprocessing by turning labels into binaries (benign is 1)
# We are doing a "double conversion" to convert everything to Binary type
for count, value in enumerate(y):
    if value == b"benign":
        y[count] = "yes"
    else:
        y[count] = "no"
lb = LabelBinarizer()
y = lb.fit_transform(y)

# ---------------------------------------------------- QUESTION 5 ---------------------------------------------------- #

new_df_benign = df[df["Class"] == b'benign']
new_df_malignant = df[df["Class"] == b'malignant']

fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)) = plt.subplots(nrows=3, ncols=3)

n_bins = 10

colors = ['#EB89B5', '#330C73']

x0 = [new_df_benign['Clump_Thickness'], new_df_malignant['Clump_Thickness']]
ax0.hist(x0, n_bins, density=True, histtype='bar', color=colors)
ax0.set_title('Clump Thickness')

x1 = [new_df_benign['Cell_Size_Uniformity'], new_df_malignant['Cell_Size_Uniformity']]
ax1.hist(x1, n_bins, density=True, histtype='bar', color=colors)
ax1.set_title('Cell Size Uniformity')

x2 = [new_df_benign['Cell_Shape_Uniformity'], new_df_malignant['Cell_Shape_Uniformity']]
ax2.hist(x2, n_bins, density=True, histtype='bar', color=colors)
ax2.set_title('Cell Shape Uniformity')

x3 = [new_df_benign['Marginal_Adhesion'], new_df_malignant['Marginal_Adhesion']]
ax3.hist(x3, n_bins, density=True, histtype='bar', color=colors)
ax3.set_title('Marginal Adhesion')

x4 = [new_df_benign['Single_Epi_Cell_Size'], new_df_malignant['Single_Epi_Cell_Size']]
ax4.hist(x4, n_bins, density=True, histtype='bar', color=colors)
ax4.set_title('Single Epi Cell Size')

x5 = [new_df_benign['Bare_Nuclei'], new_df_malignant['Bare_Nuclei']]
ax5.hist(x4, n_bins, density=True, histtype='bar', color=colors)
ax5.set_title('Bare Nuclei')

x6 = [new_df_benign['Bland_Chromatin'], new_df_malignant['Bland_Chromatin']]
ax6.hist(x4, n_bins, density=True, histtype='bar', color=colors)
ax6.set_title('Bland Chromatin')

x7 = [new_df_benign['Normal_Nucleoli'], new_df_malignant['Normal_Nucleoli']]
ax7.hist(x4, n_bins, density=True, histtype='bar', color=colors)
ax7.set_title('Normal Nucleoli')

x8 = [new_df_benign['Mitoses'], new_df_malignant['Mitoses']]
ax8.hist(x4, n_bins, density=True, histtype='bar', color=colors)
ax8.set_title('Mitoses')

plt.legend(['Benign', 'Malignant'], loc='best')
plt.tight_layout()
plt.show()

# ---------------------------------------------------- QUESTION 6 ---------------------------------------------------- #

# We need to create a classifier for each number of neighbours
for n in NEIGHBOURS:

    print(f"Classifying n = {n}:")

    # Creates a k fold cross validator
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=GROUP_NUMBER)

    # Creates KNN classifier for n neighbours
    clf = KNeighborsClassifier(n, weights="uniform", p=2, metric="minkowski")

    # For each train/test set, we use a KNN classifier
    for train_index, test_index in skf.split(X, y):

        # Uses indexes to fetch which values are going to be used to train and test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Trains knn classifier
        clf.fit(X_train, y_train.ravel())

        # Uses testing data and gets model accuracy
        acc = clf.score(X_test, y_test)
        print(f"Acc using test data {acc}")

        # Uses training data and gets model training accuracy to determine and analise over-fitting
        acc = clf.score(X_train, y_train)
        print(f"Acc using training data {acc}")

    print("\n")

# ---------------------------------------------------- QUESTION 7 ---------------------------------------------------- #


# ---------------------------------------------------- QUESTION 8 ---------------------------------------------------- #
