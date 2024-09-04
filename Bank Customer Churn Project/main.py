from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


train_df = spark.read.parquet("train_data.parquet")
test_df = spark.read.parquet("test_data.parquet")

lr = LogisticRegression(featuresCol="features", labelCol="label")
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100)
gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=10)

#Training
lr_model = lr.fit(train_df)
rf_model = rf.fit(train_df)
gbt_model = gbt.fit(train_df)

#Preds
lr_predictions = lr_model.transform(test_df)
rf_predictions = rf_model.transform(test_df)
gbt_predictions = gbt_model.transform(test_df)

#Evaluating
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

lr_auc = evaluator.evaluate(lr_predictions)
rf_auc = evaluator.evaluate(rf_predictions)
gbt_auc = evaluator.evaluate(gbt_predictions)

print(f"Logistic Regression AUC: {lr_auc}")
print(f"Random Forest AUC: {rf_auc}")
print(f"Gradient Boosted Trees AUC: {gbt_auc}")

# Save Random Forest - performed best
best_model = rf_model
best_model.write().overwrite().save("best_model")