# 仅用于学习日记 不可用于任何商业用途
#目的：利用GBDT对客户流失进行预测
#1、整理数据
#利用字典的形式，将数据中的的非数字转为数字，如为空转为0
#2、分割数据
#利用sklearn中的train_test_split(data,test_size,random_state)
#3、定义模型和训练
#model=sklearn.ensemble.GradientBoostingClassifier(learning_rate,n_estimators,max_dapth)
#model.fit(train_x,train_y)
#4、评估模型
#model.predict(test_x)预测y
#预测出的y为概率  所以需要二分类  y>0.5 则为1 反之为0
#sklearn.metrics.mean_squared_error(y_predict,y_test)计算mse
#sklearn.metrics.accuracy_score(y_predict,y_test) 计算正确率
