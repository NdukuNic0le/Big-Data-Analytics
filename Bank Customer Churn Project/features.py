from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

df = spark.read.parquet("customer_data.parquet")


categorical_cols = ["job", "marital", "education", "credit_default", "has_loan", "has_credit_card"]
numerical_cols = ["age", "balance", "transaction_count", "total_withdrawals", "total_deposits"]

stages = []


for col in categorical_cols:
    indexer = StringIndexer(inputCol=col, outputCol=f"{col}_indexed")
    encoder = OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded")
    stages += [indexer, encoder]


assembler_inputs = [f"{col}_encoded" for col in categorical_cols] + numerical_cols
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
stages += [assembler]

pipeline = Pipeline(stages=stages)
model = pipeline.fit(df)
df_vectorized = model.transform(df)

# Selecting the final columns for modeling
final_df = df_vectorized.select("features", F.col("churned").alias("label"))

train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)
train_df.write.parquet("train_data.parquet")
test_df.write.parquet("test_data.parquet")


model.write().overwrite().save("features_pipeline")