from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
from pyspark.ml.feature import PCA
from pyspark.ml import Pipeline


spark = SparkSession.builder.appName("FinancialForecasting").getOrCreate()

df = spark.read.csv("UCI.csv", header=True, inferSchema=True)

feature_cols = ["X6", "X7", "X8", "X9","fico_range_low", "dti", "pub_rec", "total_acc"]

# feature vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# PCA for dimensionality reduction
pca = PCA(k=3, inputCol="scaled_features", outputCol="pca_features")

# Models
rf = RandomForestClassifier(labelCol="default", featuresCol="pca_features", numTrees=10)
gbt = GBTClassifier(labelCol="default", featuresCol="pca_features", maxIter=10)


# Create pipelines
pipeline_rf = Pipeline(stages=[assembler, scaler, pca, rf])
pipeline_gbt = Pipeline(stages=[assembler, scaler, pca, gbt])

# Parameter grids for hyperparameter tuning
paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

paramGrid_gbt = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [10, 20, 30]) \
    .addGrid(gbt.maxDepth, [5, 10, 15]) \
    .build()

# Define evaluator
evaluator = BinaryClassificationEvaluator(labelCol="default", metricName="areaUnderROC")

# Create CrossValidator for each model
cv_rf = CrossValidator(estimator=pipeline_rf,
                       estimatorParamMaps=paramGrid_rf,
                       evaluator=evaluator,
                       numFolds=3)

cv_gbt = CrossValidator(estimator=pipeline_gbt,
                        estimatorParamMaps=paramGrid_gbt,
                        evaluator=evaluator,
                        numFolds=3)

# Fit the models
model_rf = cv_rf.fit(df)
model_gbt = cv_gbt.fit(df)

# predictions
predictions_rf = model_rf.transform(df)
predictions_gbt = model_gbt.transform(df)

# Evaluation
auc_rf = evaluator.evaluate(predictions_rf)
auc_gbt = evaluator.evaluate(predictions_gbt)

print(f"Random Forest AUC: {auc_rf}")
print(f"Gradient Boosted Trees AUC: {auc_gbt}")

# Performance Tuning: Caching and Persistence
df.cache()

# Get the best model and its parameters
best_model_rf = model_rf.bestModel
best_params_rf = {param.name: value for param, value in best_model_rf.extractParamMap().items() if param.name in ['numTrees', 'maxDepth']}
print("Best Random Forest Parameters:", best_params_rf)
best_model_gbt = model_gbt.bestModel
best_params_gbt = {param.name: value for param, value in best_model_gbt.extractParamMap().items() if param.name in ['maxIter', 'maxDepth']}
print("Best Gradient Boosted Trees Parameters:", best_params_gbt)

# Feature Importance (for Random Forest)
feature_importance = best_model_rf.stages[-1].featureImportances
for i, imp in enumerate(feature_importance):
    print(f"Feature {feature_cols[i]} importance: {imp}")

# Clean up
spark.stop()