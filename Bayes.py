import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import json
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB


class Bayes:

	def read(self):
		with open('./data.json','r',encoding='utf8') as f:
			json_data = json.load(f)
			data_=json_data['data']
			target_=json_data['target']
			X_train, X_test, y_train, y_test=train_test_split(data_,target_,
							 test_size=0.2,
							 random_state=1)
			return X_train, X_test, y_train, y_test

	def bayes(self):
		X_train, X_test, y_train, y_test=self.read()
		clf = GaussianNB()
		clf = clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		print("高斯朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (len(X_test), (y_test != y_pred).sum()))
		y_pre = []
		for X in X_test:
			y_pre.append(clf.predict_proba([X])[0][0])
		return y_pre,y_test

	def roc(self):
		y_pre, y_test=self.bayes()
		fpr, tpr, thersholds = roc_curve(y_test, y_pre, pos_label=0)
		for i, value in enumerate(thersholds):
			print("%f %f %f" % (fpr[i], tpr[i], value))
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
		plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
		plt.ylim([-0.05, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
		plt.title('ROC Curve')
		plt.legend(loc="lower right")
		plt.show()


if __name__ == '__main__':
	bay=Bayes()
	bay.roc()
