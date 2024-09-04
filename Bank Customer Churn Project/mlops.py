import mlflow
import mlflow.spark
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import current_timestamp

mlflow.set_experiment("bank_churn_prediction")

feature_pipeline = Pipeline.load("features_pipeline")
tuned_model = CrossValidatorModel.load("tuned_model")
full_pipeline = Pipeline(stages=feature_pipeline.getStages() + [tuned_model.bestModel])

def predict_churn(batch_df):
    #timestamp for tracking
    batch_df = batch_df.withColumn("prediction_timestamp", current_timestamp())
    predictions = full_pipeline.transform(batch_df)
    output = predictions.select("customer_id", "prediction", "prediction_timestamp")
    mlflow.spark.log_model(full_pipeline, "churn_prediction_model")
    mlflow.log_metric("batch_size", batch_df.count())
    
    return output

#Simulating batch prediction
new_data = spark.read.jdbc(url=jdbc_url, table="new_customers", properties=connection_properties)
results = predict_churn(new_data)

results.write.mode("overwrite").parquet("churn_predictions")

#model monitoring
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, CatTargetDriftTab

reference_data = spark.read.parquet("train_data.parquet").toPandas()
current_data = new_data.toPandas()

dashboard = Dashboard(tabs=[DataDriftTab, CatTargetDriftTab])
dashboard.calculate(reference_data, current_data, column_mapping=None)
dashboard.save("model_monitoring_report.html")