import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import json
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import confusion_matrix

class Decision_Tree:

	def read(self):
		with open('./data.json','r',encoding='utf8') as f:
			json_data = json.load(f)
			data_=json_data['data']
			target_=json_data['target']
			X_train, X_test, y_train, y_test=train_test_split(data_,target_,
							 test_size=0.2,
							 random_state=1)
			return X_train, X_test, y_train, y_test

	def tree_classes(self):
		X_train, X_test, y_train, y_test=self.read()
		clf = tree.DecisionTreeClassifier()
		clf = clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		print("决策树分类，样本总数： %d 错误样本数 : %d" % (len(X_test), (y_test != y_pred).sum()))
		return y_pred,y_test

	def matrix(self):
		y_pred, y_test=self.tree_classes()
		C = confusion_matrix(y_test,y_pred, labels=[0,1])  # 可将'1'等替换成自己的类别，如'cat'。
		plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
		# plt.colorbar()
		for i in range(len(C)):
			for j in range(len(C)):
				plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
		# plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		# plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
		# plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
		plt.show()

if __name__ == '__main__':
	bay=Decision_Tree()
	bay.matrix()
