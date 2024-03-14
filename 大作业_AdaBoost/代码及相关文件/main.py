# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:12:31 2023

@author: Chenxiakun
"""
import numpy as np
import pandas as pd

base = 0  # 选择基分类器

# 决策树桩预测模型
class Decision_Tree():
    # 对单一特征单一阈值进行分类
    def DT_stump_classify(self, data, feature, threshold, method):
        y_predict = np.ones(shape=(data.shape[0], 1))  # 初始化所有样本的分类结果为+1
        if method == 'lt':
            y_predict[data[:, feature] <= threshold] = -1.0  # 将第dimen个特征小于threshold的样本标记为-1
        else:
            y_predict[data[:, feature] > threshold] = 1.0  # 将第dimen个特征大于threshold的样本标记为+1
        return y_predict
     
    # 通过遍历所有的特征，每个特征取不同的阈值进行测试，找到分类效果最佳的特征和阈值
    def DT_build_stump(self, data, label, weights, learning_rate):
        data_mat = np.mat(data)
        label_mat = np.mat(label)
        label_mat[label_mat == 0] = -1
        m, n = data_mat.shape
        y_predict = np.mat(np.zeros(shape=(m, 1)))
        min_error = float('inf')
        for i in range(n):  # 遍历所有的特征
            unique_feature = np.unique(data[:, i])
            for j in range(len(unique_feature)):    # 遍历第i个特征下的所有不同值，尝试将其（+0.001）作为阈值
                for method in ['lt', 'gt']:  # 遍历所有判断方法，大于阈值赋1或者小于阈值赋1
                    threshold = unique_feature[j] + 0.001 
                    # 计算在当前分支阈值条件下，决策树的分类结果
                    y_predict_temp = self.DT_stump_classify(data_mat, i, threshold, method)  
                    # error矩阵用于保存决策树的预测结果
                    error = np.mat(np.ones(shape=(m, 1))) 
                    # 将error矩阵中被当前决策树分类正确的样本对应位置的值置为0
                    error[y_predict_temp == label_mat] = 0   
                    # 计算分类错误率（按位相乘并求和），错误率=所有分类错误样本的权重求和
                    weights_error = weights.T * error  
                    # 如果误差率降低，保存最佳分类方法的相关信息
                    if weights_error < min_error:  
                        #print('i的值是：',i)
                        min_error = weights_error    
                        y_predict = y_predict_temp.copy()  
                        best_feature = i  
                        best_threshold = threshold  
                        best_method = method  
        return best_feature, best_threshold, best_method, min_error, y_predict
    
# 对数回归预测模型
class Logistic_Regression(): 
    # 初始化
    def LR_init(self, data, label):
        self.theta = np.zeros((data.shape[1] + 1, 1))   # LR和AdaBoost是两个类，如果这句定义在train函数里，每次运行train都会初始化theta的值
        # 这样即使数据的weights正常更新，算法步骤之类的也没有错，theta也会几乎没有变化，从而导致弱分类器的性能没有得到提升
        self.w = np.ones((data.shape[1], 1)) # w是一个大小为[data.shape[1], 1]的新元组
        self.b = 0
        
    # 对数几率函数
    def LR_sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
        
    # 梯度计算函数（注释掉的函数都是弃用的）
    #def LR_gradient(self, data, label, w, b, weights):
    #    y_sigmoid = self.LR_sigmoid(np.dot(data, w) + b)
    #    temp = weights * (label - y_sigmoid) # 不乘个weights的话，每一轮训练出来的w和b是不会变的
    #    #temp = y_sigmoid - label   # 与LR_train函数中两个-=相对应
    #    #temp = label - y_sigmoid  # 与LR_train函数中两个+=相对应
    #    #dw = np.dot(data.T, temp) / data.shape[0]  # 正确率低
    #    #db = np.sum(temp) / data.shape[0] # 正确率低
    #    dw = np.dot(data.T, temp)
    #    db = np.sum(temp)  # dw、db的计算公式是由损失函数推导出来的
    #    return dw, db
    
    #def LR_gradient2(self, data, label, theta, weights):
     #   y_sigmoid = self.LR_sigmoid(np.dot(data, theta))
      #  temp = weights * (label - y_sigmoid)
       # return np.dot(data.T, temp)
    
    # 对数回归训练模型
    #def LR_train(self, data, label, weights, learning_rate = 0.01, train_round =8000):
    #    learning_rate = 0.03
    #    for i in range(train_round):
    #        y_sigmoid = self.LR_sigmoid(np.dot(data, self.w) + self.b)
    #        error = weights * (label - y_sigmoid)
    #        dw = np.dot(data.T, error)
    #        db = np.sum(error)
    #        #dw, db = self.LR_gradient(data, label, self.w, self.b, weights)
    #        # 梯度下降计算新的w、b值
    #        self.w += learning_rate * dw
    #        self.b += learning_rate * db
    #        #w -= learning_rate * dw
    #        #b -= learning_rate * db
    #    #print(w)
    #    return self.w, self.b   
    
    # 对数回归预测
    #def LR_predict(self, data, w, b):
    #    y_sigmoid = self.LR_sigmoid(np.dot(data, w) + b)
    #    y_predict = np.array([1 if y_sigmoid[i] > 0.5 else 0 for i in range(len(y_sigmoid))])
    #    return y_predict
    
    def LR_train2(self, data, label, weights, learning_rate, train_round =9000):
        learning_rate = 0.03
        row, col = data.shape
        #theta = np.zeros((col + 1, 1))
        data_expan = np.c_[data, np.ones(row)]  #c_函数：列连接
        for i in range(train_round):
            y_sigomid = self.LR_sigmoid(np.dot(data_expan, self.theta))
            error = label - y_sigomid
            self.theta += learning_rate * (np.dot(data_expan.T, weights * error))
        return self.theta 
    
    def LR_predict2(self, data, theta):
        data_expan = np.c_[data, np.ones(data.shape[0])]
        y_sigmoid = self.LR_sigmoid(np.dot(data_expan, theta))
        #print(theta.shape)
        y_sigmoid[y_sigmoid > 0.5] = 1
        y_sigmoid[y_sigmoid <= 0.5] = 0
        return y_sigmoid


# Adaboost算法
class Adaboost():
    # 初始化参数
    def init(self, base, data, label, n):
        self.base = base
        self.clf_sets = []  # 弱分类器集合
        self.alpha = []     # 弱分类器的alpha
        self.clf_num = n    # 弱分类器的个数
        self.row, self.col = data.shape
        self.weights = np.ones((self.row, 1)) / self.row  # 初始化各数据权重，初始值为：1/数据个数
        #print(self.weights.shape)
        
    # 训练模型
    def train(self, data, label, learning_rate, n, base):
        # 初始化
        self.init(base, data, label, n)
        if self.base == 0: # 对数回归
            lr = Logistic_Regression()
            lr.LR_init(data, label)
            for train_round in range(self.clf_num):   
                print('对数几率回归：第', train_round + 1, '个分类器，is training')
                #w, b = lr.LR_train(data, label, self.weights, learning_rate)  # 得到一组弱分类器
                theta = lr.LR_train2(data, label, self.weights, learning_rate)
                #y_predict = lr.LR_predict(data, w, b)
                y_predict = lr.LR_predict2(data, theta)
                # 将弱分类器加入集合
                #self.clf_sets.append((w, b))
                self.clf_sets.append(theta)
                e = self.cal_e(y_predict, label)
                print('错误率：', e)
                alpha = self.cal_alpha(e)
                # 将α值加入集合
                self.alpha.append(alpha)
                # 计算规范化因子
                Z = self.cal_Z(self.weights, alpha, label, y_predict)
                # 更新权重
                self.update_weights(Z, alpha, label, y_predict)
                #self.weights = self.weights / self.weights.sum()
                #print(self.weights[0:9])
        else:   # 决策树
            dt = Decision_Tree()
            for train_round in range(self.clf_num):
                print('决策树：第', train_round + 1, '个分类器，is training')
                decisive_feature, threshold, method, e, y_predict = dt.DT_build_stump(data, label, self.weights, learning_rate)
                print('决定特征编号：', decisive_feature)
                self.clf_sets.append((decisive_feature, threshold, method))
                print('错误率：', e)
                alpha = self.cal_alpha(e)
                # 将α值加入集合
                self.alpha.append(alpha)
                # 计算规范化因子
                Z = self.cal_Z(self.weights, alpha, label, y_predict)
                # 更新权重
                self.update_weights(Z, alpha, label, y_predict)
                
            
    
    # 计算弱分类器的误差率       
    def cal_e (self, y_predict, label):
        return sum([self.weights[i] for i in range(len(label)) if y_predict[i] != label[i]])
    
    # 计算弱分类器的α值
    def cal_alpha(self, e):
        return 0.5 * np.log((1 - e) / max(e, 1e-16))
    
    # 计算规范化因子
    def cal_Z(self, weights, alpha, y_real, y_predict):
        return sum([weights[i] * np.exp(-1 * alpha * y_real[i] * y_predict[i]) for i in range(len(y_real))])
    
    # 更新权重
    def update_weights(self, Z, alpha, y_real, y_predict):
        for i in range(len(y_real)):
            #self.weights[i] = (self.weights[i] * np.exp(-1 * alpha * y_real[i] * y_predict[i])) / Z
            self.weights[i] = self.weights[i] * np.exp(-1 * alpha * y_real[i] * y_predict[i])
        self.weights = self.weights / self.weights.sum() # 权重归一化
    
    # adaboost的预测函数    
    def predict(self, data):
        y_temp = np.zeros((data.shape[0], 1))
        if (self.base == 0):    # 对数回归预测
            lr = Logistic_Regression()
            for i in range(len(self.clf_sets)):
                #print(len(self.clf_sets))
                #w, b = self.clf_sets[i]
                theta = self.clf_sets[i]
                alpha = self.alpha[i]
                print('alpha: ',alpha)  
                #y_predict = lr.LR_predict(np.mat(data), w, b)
                y_predict = lr.LR_predict2(np.mat(data), theta)
                y_predict[y_predict == 0] = -1
                for j in range(data.shape[0]):
                    y_temp[j] = y_temp[j] + alpha * y_predict[j]
            #print(y_temp)
            y_final = np.array([1 if y_temp[i] > 0 else 0 for i in range(data.shape[0])])
        else:   # 决策树预测
            dt = Decision_Tree()
            for i in range(len(self.clf_sets)):
                decisive_feature, threshold, method = self.clf_sets[i]
                alpha = self.alpha[i]
                #print('alpha: ', alpha)
                y_predict = dt.DT_stump_classify(np.mat(data), decisive_feature, threshold, method)
                for j in range(data.shape[0]):
                    y_temp[j] = y_temp[j] + alpha * y_predict[j]
            y_final = np.array([1 if y_temp[i] > 0 else 0 for i in range(data.shape[0])])
        return y_final
        
# 数据预处理：归一化
def min_max_normalization(data):
    data_row, data_col = np.shape(data)
    for j in range(data_col):
        col_mindata = min(data[:,j])
        col_maxdata = max(data[:,j])
        for i in range(data_row):
            data[i,j] = (data[i,j] - col_mindata)/(col_maxdata - col_mindata)
    return data

# 数据预处理：均值归一化
def mean_normalization(data):
    data_row, data_col = np.shape(data)
    for j in range(data_col):
        col_mindata = min(data[:,j])
        col_maxdata = max(data[:,j])
        col_averdata = np.mean(data[:,j])
        for i in range(data_row):
            data[i,j] = (data[i,j] - col_averdata)/(col_maxdata - col_mindata)
    return data

# 十折交叉验证生成数据样本
# 数据分为十等份，一份作test，九份作train
def ten_fold_split(data, label, fold):
    data_row, data_col = np.shape(data)
    #print(data_row, data_col)   # 3680 57
    data_row = int(data_row)
    data_col = int(data_col)
    test_len = int(data_row / 10)
    #print(test_len)
    train_data = np.zeros((data_row - test_len, data_col))  # zeros函数：创建指定大小且值全为0的数组
    train_label = np.zeros((data_row - test_len, 1))
    test_data = np.zeros((test_len, data_col))
    test_label = np.zeros((test_len, 1))
    j = 0
    for i in range(10):
        if i == fold - 1:
            test_data = data[i * test_len : (i+1) * test_len, :]
            test_label = label[i * test_len : (i+1) * test_len]
        else:
            train_data[j * test_len : (j+1) * test_len, :] = data[i * test_len :(i+1) * test_len, :]
            train_label[j * test_len : (j+1) * test_len] = label[i * test_len : (i+1) * test_len]
            j = j + 1
    return test_len, train_data, train_label, test_data, test_label


# 主函数体
if __name__ == '__main__':
    data = np.array(pd.read_csv('data.csv', header = None))    # array函数：列表转换为数组
    #print(data[3679]) 
    label = np.array(pd.read_csv('targets.csv', header = None))
    # 数据预处理
    #data = min_max_normalization(data) # 作用极大(对数回归最高0.8423，决策树最高0.878)
    # 不加预处理的话，对数回归正确率又会越来越低
    data = mean_normalization(data)#（对数回归最高0.8429，决策树最高0.879）
    base_list = [1, 5, 10, 100]  # 基分类器个数
    ada = Adaboost()
    
    # 学习率
    learning_rate = 0.01
    # 以下是十折交叉验证的代码，预测test.csv时请将下方这段代码注释掉 
    '''
    for base_num in base_list:
       for fold in range(1, 11):
            # 十折交叉验证生成数据样本
            test_len, train_data, train_label, test_data, test_label = ten_fold_split(data, label, fold)
            ada.train(train_data, train_label, learning_rate, base_num, base)# 训练模型
            #ada.train(data, label, learning_rate, base_num, base)
            #print(ada.alpha)
            predict_result = ada.predict(test_data)# 数据预测（分类）
            # 创建空数组
            save_result = np.zeros((len(predict_result), 2), dtype = int)
            # 第一列：样例的序号
            save_result[:, 0] = np.linspace(test_len * (fold - 1) + 1, test_len * fold, test_len)  # linspace函数，生成等差数列，三个参数分别是起始值、结束值、生成的数字数量
            # 第二列：预测结果
            save_result[:, 1] = predict_result
            # 计算并输出精度值
            accuracy = 0
            for k in range(len(test_label)):
                if predict_result[k] == test_label[k]:
                    accuracy += 1
            accuracy /= len(test_label)
            print(base_num, "个基分类器,第", fold, "折验证，正确率为：", accuracy)
            # 保存到文件
            np.savetxt('experiments/base%d_fold%d.csv' % (base_num, fold), save_result, fmt='%d', delimiter=',')
    '''    
    # 十折交叉验证代码结束
    
    # 以下是为了预测test.csv文件，新增加的内容，十折交叉验证时请将下方这段代码注释掉        
    base_num_new = 100; # 基分类器数目
    test_data_new = np.array(pd.read_csv('test.csv', header = None))
    test_data_new = mean_normalization(test_data_new)    # 数据预处理
    #test_label_new = np.array(pd.read_csv('test_label.csv', header = None))   # 自己测试用
    row_new, col_new = np.shape(test_data_new)    # row_new代表了数据量
    ada.train(data, label, learning_rate, base_num_new, base) # 用全部数据训练一组基分类器
    predict_result_new = ada.predict(test_data_new)
    save_result_new = np.zeros((len(predict_result_new), 2), dtype = int)
    save_result_new[:,0] = np.linspace(1, row_new, row_new)
    save_result_new[:,1] = predict_result_new
    # 自己测试用输出正确率
    #accuracy = 0
    #for k in range(row_new):
    #    if predict_result_new[k] == test_label_new[k]:
    #        accuracy += 1
    #accuracy /= row_new    
    #print("正确率为：", accuracy)
    np.savetxt('predict_y.csv', save_result_new, fmt = "%d", delimiter = ',')