import findspark
import pyspark
from pyspark.mllib.tree import RandomForest
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.sql.session import SparkSession
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def create_labeled_point(spCont, features, labels, categorical=False):
    points = []
    for i, j in zip(features, labels):
        lp = LabeledPoint(j, i)
        points.append(lp)
    return spCont.parallelize(points)


findspark.init()
findspark.find()


config = pyspark.SparkConf().setAppName('wine-predict').setMaster('local')
sparkContext = pyspark.SparkContext(conf=config)
spark = SparkSession(sparkContext)



dataFrame = spark.read.format("csv").load("TrainingDataset.csv", header=True, sep=";")
dataFrame.printSchema()
dataFrame.show()


for column in dataFrame.columns[1:-1] + ['""""quality"""""']:
    dataFrame = dataFrame.withColumn(column, col(column).cast('float'))
dataFrame = dataFrame.withColumnRenamed('""""quality"""""', "label")


features = np.array(dataFrame.select(dataFrame.columns[1:-1]).collect())
label = np.array(dataFrame.select('label').collect())


#vector assembler and transformation
VectorAssembler = VectorAssembler(inputCols=dataFrame.columns[1:-1], outputCol='features')
transformedDf = VectorAssembler.transform(dataFrame)
transformedDf = transformedDf.select(['features', 'label'])


#labeled points
sample = create_labeled_point(sparkContext, features, label)


#Split
train, test = sample.randomSplit([0.7, 0.3], seed=11)


#Creating classifier
model = RandomForest.trainClassifier(train, numClasses=10, categoricalFeaturesInfo={},
                                     numTrees=21, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=30, maxBins=32)


#predicting result
predictedLabel = model.predict(test.map(lambda x: x.features))


#mapping the actual labels and predicted labels
labelDiff = test.map(lambda lp: lp.label).zip(predictedLabel)


labelDiffDf = labelDiff.toDF()
labelPredict = labelDiff.toDF(["label", "Prediction"])
labelPredict.show()
labelpred_df = labelPredict.toPandas()


#F1-score calculation
F1score = f1_score(labelpred_df['label'], labelpred_df['Prediction'], average='micro')
print("F1-score: ", F1score)
print(confusion_matrix(labelpred_df['label'], labelpred_df['Prediction']))
print(classification_report(labelpred_df['label'], labelpred_df['Prediction']))
print("Accuracy:", accuracy_score(labelpred_df['label'], labelpred_df['Prediction']))

#test error calculation
testError = labelDiff.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test.count())
print('Test Error = ' + str(testError))

#model.save(sparkContext, 's3://winequalitypred/output/')

model.save(sparkContext, "wine_quality_model")

