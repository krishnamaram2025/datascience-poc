import pyspark
from pyspark.mllib.tree import RandomForestModel
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.sql.session import SparkSession
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def create_labeled_pts(sparkContext, features, labels):
    points = []
    for i, j in zip(features, labels):
        lp = LabeledPoint(j, i)
        points.append(lp)
    return sparkContext.parallelize(points)



config = pyspark.SparkConf().setAppName('winequality').setMaster('local')
sparkContext = pyspark.SparkContext(conf=config)
spark = SparkSession(sparkContext)


#dataFrame = spark.read.format("csv").load("./ValidationDataset.csv", header=True, sep=";")

dataFrame = spark.read.format("csv").load("ValidationDataset.csv", header=True, sep=";")
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
sample = create_labeled_pts(sparkContext, features, label)

# loads model
#model = RandomForestModel.load(sparkContext, "./model/")

model = RandomForestModel.load(sparkContext, "wine_quality_model")
print("Model loaded!")

#predicting result
predictedLabels = model.predict(sample.map(lambda x: x.features))

#mapping the actual labels and predicted labels
difference = sample.map(lambda lp: lp.label).zip(predictedLabels)

differenceDf = difference.toDF()
labelPredicted = difference.toDF(["label", "Prediction"])
labelPredicted.show()
labelPredictedDf = labelPredicted.toPandas()

#F1-score calculation
F1score = f1_score(labelPredictedDf['label'], labelPredictedDf['Prediction'], average='micro')
print("F1-score: ", F1score)
print(confusion_matrix(labelPredictedDf['label'], labelPredictedDf['Prediction']))
print(classification_report(labelPredictedDf['label'], labelPredictedDf['Prediction']))
print("Accuracy", accuracy_score(labelPredictedDf['label'], labelPredictedDf['Prediction']))

#test error calculation
testError = difference.filter(
    lambda lp: lp[0] != lp[1]).count() / float(sample.count())
print('Test Error = ' + str(testError))
