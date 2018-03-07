"""
Description:
    auxiliary functions
"""
from Toolbox.Preprocessing import Processor
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import maxabs_scale
from sklearn.metrics import accuracy_score


def eval_band(new_img, gt, train_inx, test_idx):
    p = Processor()
    # img_, gt_ = p.get_correct(new_img, gt)
    gt_ = gt
    img_ = maxabs_scale(new_img)
    # X_train, X_test, y_train, y_test = train_test_split(img_, gt_, test_size=0.4, random_state=42)
    X_train, X_test, y_train, y_test = img_[train_inx], img_[test_idx], gt_[train_inx], gt_[test_idx]
    knn_classifier = KNN(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    # score = cross_val_score(knn_classifier, img_, y=gt_, cv=3)
    y_pre = knn_classifier.predict(X_test)
    score = accuracy_score(y_test, y_pre)
    # score = np.mean(score)
    return score