## 安然提交开放式问题

@(Machine Learning)


### Q&A:
__Q1: *向我们总结此项目的目标以及机器学习对于实现此目标有何帮助。作为答案的部分，提供一些数据集背景信息以及这些信息如何用于回答项目问题。你在获得数据时它们是否包含任何异常值，你是如何进行处理的？【相关标准项：“数据探索”，“异常值调查”】*__

该项目的目标是使用安然财务造假一案中的邮件数据来设计预测模型来找出作案嫌疑人(POI)，由于该案已经结束，犯罪分子也已经被捕获，所以这份数据就可以很好的被作为研究案例，从而设计出好的预测模型运用到潜在犯罪公司的识别中。

项目中提供的数据集包含email数据和financial数据，一共有146条可用数据，每条数据代表一个人，其中每个人有20个特征值来表示，其中有35个被标记为POI的人，但在最终的项目数据中146个人被标记为POI的人只有18人，通过`./enron61702insiderpay`，可以发现数据集中各个特征值的缺失情况，如下：

| 特征 | 缺失个数 |
| :--- | :--: |
| Loan advances | 142 |
| Director fees | 129 |
| Restricted stock deferred | 128 |
| Deferred payment | 107 |
| Deferred income | 97 |
| Long term incentive | 80 |
| Bonus | 64 |
| Emails sent also to POI | 60 |
| Emails sent | 60 |
| Emails received | 60 |
| Emails from POI | 60 |
| Emails to POI | 60 |
| Other | 53 |
| Expenses | 51 |
| Salary | 51 |
| Excersised stock option | 44 |
| Restricted stock | 36 |
| Email address | 35 |
| Total payment | 21 |
| Total stock value | 20 |

从上面的数据可以看出：

+ 这个数据集很不平衡（imbalance）, 也就说明accuracy并不是很好的评估指标，所以在后面模型评估环节会使用precision和recall。
+ 在交叉验证的时候，因为数据的不平衡性（POI的人数较样本量偏少），所以打算选用Stratified Shuffle Split的方式将数据分为验证集和测试集。

因为在大部分情况下，最好的特征都来自于我们的直觉，所以对于安然欺诈案稍微交代背景也是很有必要的。安然公司曾经是世界上最大的能源、商品和服务公司之一，名列《财富》杂志“美国500强”的第七名，然而，2001年12月2日，安然公司突然向纽约破产法院申请破产保护，该案成为美国历史上企业第二大破产案。详情可点击[安然事件及其启示](http://www.360doc.com/content/11/0531/19/2471108_120782588.shtml)。

本项目将主要使用`scikit-learn` 软件包及一些机器学习的方法来预测嫌疑人(POI)，期间会使用到每一个用户特征中的金融及邮件信息。通过数据集探索，将删除以下异常值:

- `TOTAL`: 由于该数据其实是每一个financial数据的汇总，所以是异常值，需要去除。
- `THE TRAVEL AGENCY IN THE PARK`: 从该记录的命名来看完全属于异常值，去除掉。
- `LOCKHART EUGENE E`: 这条记录没有任何特征信息

__Q2: *你最终在你的 POI 标识符中使用了什么特征，你使用了什么筛选过程来挑选它们？你是否需要进行任何缩放？为什么？作为任务的一部分，你应该尝试设计自己的特征，而非使用数据集中现成的——解释你尝试创建的特征及其基本原理。（你不一定要在最后的分析中使用它，而只设计并测试它）。在你的特征选择步骤，如果你使用了算法（如决策树），请也给出所使用特征的特征重要性；如果你使用了自动特征选择函数（如 SelectBest），请报告特征得分及你所选的参数值的原因。【相关标准项：“创建新特征”、“适当缩放特征”、“智能选择功能”】*__

在最终的POI标识识别时，使用了如下表所示的10个特征，是通过 scikit-learn的 `SelectKBest` 选出的最影响结果的10个特征。在备选的19个特征中，还通过比例缩放设计了两个特征`fraction_from_poi` 和 `fraction_to_poi`，分别代表来自poi的邮件占所有收到邮件的占比，和发给poi的邮件占所有发送邮件的占比。*<font color='red'>但使用 Select 10 Best时，并未筛选出这两个变量，`fraction_from_poi` 和 `fraction_to_poi`的得分都不到1，可能是由于POI一般属于较高层员工，每日处理邮件的数据很大，来往通讯人也很多，所以属于POI的比例不一定高。</font>*虽然`loan_advances`的缺失值非常多，但也是最重要的10个特征之一。

| Selected Features        | Score   | Percent of NAN |
| :----------------------: | :-----:  | :-----------: |
| exercised_stock_options | 24.815080 |  29.4 |
| total_stock_value       | 24.182899 |  12.6 |
| bonus                   | 20.792252 |  43.4 |
| salary                  | 18.289684 |  34.3 |
| deferred_income         | 11.458477 |  66.4 |
| long_term_incentive     |  9.922186 |  54.5 |
| restricted_stock        |  9.212811 |  23.8 |
| total_payments          |  8.772778 |  14.0 |
| shared_receipt_with_poi |  8.589421 |  39.9 |
| loan_advances           |  7.184056 |  97.9 |

__Q3: *你最终使用了什么算法？你还尝试了其他什么算法？不同算法之间的模型性能有何差异？【相关标准项：“选择算法”】*__

我首先尝试了一下GaussianNB，因为在模板中有提供该方法，得到一个准确率和召回率，然后我就以此为基准又尝试了4种算法，综合最好的是逻辑回归算法(Logistic Regression)，因其在准确率和召回率上都超过了GaussianNB，比较均衡。期间，还尝试了支持向量机(SVM)，随机森林(Random Forest), K均值(K Means)。其结果如下表所示：

| Algorithm           | Precision |  Recall  |
| :-----------------: |:--------: |  :-----: |
| GaussianNB          | 0.341673420536 | 0.357945815296 |
| Logistic Regression | 0.362552943135 | 0.369802272727 |
| K-Means           | 0.307277095072 | 0.360421031746 |
| SVM         | 0.518227252404 | 0.225785678211 |
| Random Forest       | 0.35091547619  | 0.165298484848 |

  
__Q4: *4. 调整算法的参数是什么意思，如果你不这样做会发生什么？你是如何调整特定算法的参数的？（一些算法没有需要调整的参数 – 如果你选择的算法是这种情况，指明并简要解释对于你最终未选择的模型或需要参数调整的不同模型，例如决策树分类器，你会怎么做）。【相关标准项：“调整算法”】*__ 

调整参数就是在模型训练时调整各个算法的参数使得算法更加适合所研究的问题。如果参数调整的不好会使整个算法效果差，同时也会使本应得到良好效果的算法无效化。所以在我尝试的几种机器学习算法中，尽可能的调整参数来达到更好的效果。其中`GaussianNB`分类并不需要调整参数，仅需要选择好特征值就可以来预测分类结果，也被我当做基准参考。
*<font color='red'>对于Logistic regression，主要调整了参数C和max_iter，C的值从1e-10尝试到1，每次尝试数量级加一，max_iter从50尝试到100，每次尝试迭代次数加10，最终效果如下：</font>*

```
LogisticRegression(C=1e-08, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=42, solver='liblinear', tol=0.001,
          verbose=0, warm_start=False))])

```
*<font color='red'>对于K-means，主要调整了参数init和max_iter，init尝试了k-means++和random，max_iter从100尝试到300，每次尝试迭代次数加50，最终效果如下：</font>*

```
KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=2, n_init=10,
    n_jobs=1, precompute_distances='auto', random_state=None, tol=0.001,
    verbose=0)
```

*<font color='red'>对于SVM，主要调整了参数C和gamma，C的值从1e3尝试到1e5，每次尝试值加5000，gamma从0.0001尝试到0.1，每次尝试乘以10，最终效果如下：</font>*

```
SVC(C=1000, cache_size=200, class_weight='auto', coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
  max_iter=-1, probability=False, random_state=42, shrinking=True,
  tol=0.001, verbose=False)
```

*<font color='red'>对于Random Forest，主要调整了参数max_features和max_depth，max_features的值尝试了sqrt，log2，max_depth尝试了5,6,7,8,9,10以及“auto”，最终效果如下：</font>*

```
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=5, max_features='sqrt', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=42, verbose=0, warm_start=False)
```

__Q5:*5.  什么是验证，未正确执行情况下的典型错误是什么？你是如何验证你的分析的？【相关标准项：“验证策略”】*__

验证是指使用一系列的评价指标来评估所选算法是否适应所处理问题。如果未进行合理的数据验证，会出现模型在训练集过拟合的情况，从而导致在测试集效果不佳的情况。机器学习中一个重要的话题便是模型的泛化能力，泛化能力强的模型才是好模型，对于训练好的模型，若在训练集表现差，不必说在测试集表现同样会很差，这可能是欠拟合导致；若模型在训练集表现非常好，却在测试集上差强人意，则这便是过拟合导致的，过拟合与欠拟合也可以用 Bias 与 Variance 的角度来解释，欠拟合会导致高 Bias ，过拟合会导致高 Variance ，所以模型需要在 Bias 与 Variance 之间做出一个权衡。

***解决欠拟合的方法***：
+ 增加新特征，可以考虑加入进特征组合、高次特征，来增大假设空间;
+ 尝试非线性模型，比如核SVM 、决策树、DNN等模型;
+ 如果有正则项可以较小正则项参数 λ.
+ Boosting ,Boosting 往往会有较小的 Bias，比如 Gradient Boosting 等.

***解决过拟合的方法***：
+ 交叉检验，通过交叉检验得到较优的模型参数；
+ 特征选择，减少特征数或使用较少的特征组合，对于按区间离散化的特征，增大划分的区间。
+ 正则化，常用的有 L1、L2 正则。而且 L1 正则还可以自动进行特征选择。
+ 如果有正则项则可以考虑增大正则项参数 λ.
+ 增加训练数据可以有限的避免过拟合.
+ Bagging ,将多个弱学习器Bagging 一下效果会好很多，比如随机森林等；

本项目采用`sklearn`提供的`train_test_split`来进行交叉验证，对训练集的数据分出来30%的量作为测试数据，进行1000次随机分配计算出一个平均的precision和recall。

__Q6:*6.  给出至少 2 个评估度量并说明每个的平均性能。解释对用简单的语言表明算法性能的度量的解读。【相关标准项：“评估度量的使用”】*__

本项目采用精确率&召回率进行评估。综合考虑发现获得最好效果的分类方法为逻辑回归(`precision: 0.363` & `recall: 0.370`) , 由于逻辑回归被广泛地使用在文本分类中，而本项目也涉及到邮件信息的识别，所以对于这样的结果还是比较可信的。
+ 精确率(Precision）是指在所有系统判定的“真”的样本中，确实是真的的占比，就是TP/(TP+FP)。
+ 召回率（Recall）是指在所有确实为真的样本中，被判为的“真”的占比，就是TP/(TP+FN)。

其中，将原本是真的判为真，就是TP（True Positive），原本真的判为假，就是FN（False Negative），原本假的判为真，就是FP（False Positive），原本假的判为假，就是TN（True Negative）。
本项目的精确率为0.363，表示逻辑回归预测的100个POI中，有36个为真的POI，而剩余的64个并非POI；召回率为0.370，表示该模型可以在所有的POI中找出37%。

### References:
- [Introduction to Machine Learning (Udacity)](https://www.udacity.com/course/viewer#!/c-ud120-nd)
- [交叉验证](http://www.cnblogs.com/ooon/p/5715918.html)
- [scikit-learn Documentation](http://scikit-learn.org/stable/documentation.html)
- [精确率&召回率](https://www.zhihu.com/question/30643044/answer/161955532)

### Files
- `tools/`: helper tools and functions
- `final_project/poi_id.py`: main submission file - POI identifier
- `final_project/tester.py`: Udacity-provided file, produce test result for submission


