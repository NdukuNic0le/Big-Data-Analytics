import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F


df = spark.read.parquet("customer_data.parquet")

df.describe().show()

# Correlation 
numeric_columns = [field.name for field in df.schema.fields if field.dataType.typeName() in ['double', 'int']]
correlation_matrix = df.select(numeric_columns).toPandas().corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# Churn rate 
churn_by_job = df.groupBy("job").agg(
    (F.sum("churned") / F.count("churned")).alias("churn_rate")
).orderBy("churn_rate", ascending=False)

churn_by_job_pd = churn_by_job.toPandas()
plt.figure(figsize=(12, 6))
sns.barplot(x='job', y='churn_rate', data=churn_by_job_pd)
plt.title('Churn Rate by Job')
plt.xticks(rotation=45)
plt.savefig('churn_rate_by_job.png')
plt.close()

# Balancing distribution 
plt.figure(figsize=(10, 6))
sns.histplot(data=df.toPandas(), x='balance', hue='churned', element='step', stat='density', common_norm=False)
plt.title('Balance Distribution: Churned vs Non-Churned')
plt.savefig('balance_distribution.png')
plt.close()

summary_stats = df.agg(
    F.mean("age").alias("avg_age"),
    F.mean("balance").alias("avg_balance"),
    F.mean("churned").alias("churn_rate")
).toPandas().to_dict('records')[0]

with open('summary_stats.txt', 'w') as f:
    for key, value in summary_stats.items():
        f.write(f"{key}: {value}\n")