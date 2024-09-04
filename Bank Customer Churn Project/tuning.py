from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

train_df = spark.read.parquet("train_data.parquet")
test_df = spark.read.parquet("test_data.parquet")

rf = RandomForestClassifier(featuresCol="features", labelCol="label")

paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 200]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.minInstancesPerNode, [1, 2, 4]) \
    .build()

crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=3)
cvModel = crossval.fit(train_df)

predictions = cvModel.transform(test_df)

auc = evaluator.evaluate(predictions)
print(f"Tuned Random Forest AUC: {auc}")

cvModel.write().overwrite().save("tuned_model")