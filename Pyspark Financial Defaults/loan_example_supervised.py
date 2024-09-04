from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
from pyspark.ml.feature import PCA
from pyspark.ml import Pipeline


spark = SparkSession.builder.appName("FinancialForecasting").getOrCreate()
def create_dummy_data(spark):
    data = [(1, 1000, 5, 0.02, 50000, 1),
            (0, 2000, 3, 0.03, 30000, 0),
            (1, 1500, 4, 0.025, 40000, 1),
            (0, 3000, 2, 0.04, 20000, 0)]
    columns = ["credit_score", "income", "employment_years", "debt_ratio", "loan_amount", "default"]
    return spark.createDataFrame(data, columns)

df = create_dummy_data(spark)

feature_cols = ["credit_score", "income", "employment_years", "debt_ratio", "loan_amount"]

# feature vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# PCA for dimensionality reduction
pca = PCA(k=3, inputCol="scaled_features", outputCol="pca_features")

# Models
rf = RandomForestClassifier(labelCol="loan_status", featuresCol="features", numTrees=10)
gbt = GBTClassifier(labelCol="loan_status", featuresCol="features", maxIter=10)

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

# Make predictions
predictions_rf = model_rf.transform(df)
predictions_gbt = model_gbt.transform(df)

# Evaluate models
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