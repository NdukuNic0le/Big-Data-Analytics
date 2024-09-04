from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("BankChurnPrediction") \
    .config("spark.jars", "") \
    .getOrCreate()

jdbc_url = ""
connection_properties = {
    "user": "",
    "password": "",
    "driver": ""
}

query = """
    SELECT 
        c.customer_id, 
        c.age, 
        c.job, 
        c.marital, 
        c.education, 
        c.credit_default, 
        a.balance, 
        COUNT(t.transaction_id) as transaction_count,
        SUM(CASE WHEN t.type = 'withdrawal' THEN t.amount ELSE 0 END) as total_withdrawals,
        SUM(CASE WHEN t.type = 'deposit' THEN t.amount ELSE 0 END) as total_deposits,
        p.has_loan,
        p.has_credit_card,
        CASE WHEN c.last_contact_date < DATE_SUB(CURDATE(), INTERVAL 3 MONTH) THEN 1 ELSE 0 END as churned
    FROM 
        customers c
    JOIN 
        accounts a ON c.customer_id = a.customer_id
    LEFT JOIN 
        transactions t ON a.account_id = t.account_id
    LEFT JOIN 
        products p ON c.customer_id = p.customer_id
    GROUP BY 
        c.customer_id, c.age, c.job, c.marital, c.education, c.credit_default, 
        a.balance, p.has_loan, p.has_credit_card, c.last_contact_date
"""

df = spark.read.jdbc(url=jdbc_url, table=f"({query}) as customer_data", properties=connection_properties)

df.write.parquet("customer_data.parquet")