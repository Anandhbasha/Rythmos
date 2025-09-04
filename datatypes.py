from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("RDD Example").getOrCreate()
sc = spark.sparkContext

# # # Create RDD
# # data = [1, 2, 3, 4, 5]
# # rdd = sc.parallelize(data)

# # # Transformation
# # squared = rdd.map(lambda x: x * x)

# # # Action
# # print(squared.collect())   # [1, 4, 9, 16, 25]


# # from pyspark.sql import SparkSession

# # # Create Spark Session
# spark = SparkSession.builder.appName("DataFrame Example").getOrCreate()

# # # Create DataFrame from list of tuples
# data = [("Alice", 25), ("Bob", 30), ("Cathy", 28)]
# # columns = ["Name", "Age"]

# # df = spark.createDataFrame(data, columns)

# # # Show DataFrame
# # df.show()

# # # Select a column
# # df.select("Name").show()

# # # Filter rows
# # df.filter(df.Age > 25).show()

# # # Group by
# # df.groupBy("Age").count().show()

# # spark.stop()
# from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Define schema
# schema = StructType([
#     StructField("Name", StringType(), True),
#     StructField("Age", IntegerType(), True)
# ])

# # Create DataFrame with schema
# df2 = spark.createDataFrame(data, schema)

# df2.printSchema()
# df2.show()



import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ExcelWithPandas").getOrCreate()

# Read Excel with header
pdf = pd.read_excel("students.xlsx", header=0)   # first row = header

# Convert to Spark DataFrame
df = spark.createDataFrame(pdf)

df.show()
df.printSchema()

# Read Excel into PySpark DataFrame

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


# Start Spark
spark = SparkSession.builder.appName("ExcelTransformActions").getOrCreate()

# Read Excel with pandas (needs: pip install openpyxl)
pdf = pd.read_excel("students.xlsx", header=0)

# Convert Pandas → PySpark DataFrame
df = spark.createDataFrame(pdf)

print("Original DataFrame:")
df.show()

# Transformations

df_selected = df.select("Name", "City")

# Filter students with Marks > 85
df_filtered = df.filter(col("Marks") > 85)

# Add a new column (Pass/Fail)
df_with_status = df.withColumn("Status",(col("Marks") >= 85).cast("string"))


# Actions
# Show the DataFrame (Action)
df_filtered.show()

# Collect rows into Python list (Action)
rows = df_filtered.collect()
print("Collected Rows:", rows)

# Count number of rows (Action)
count = df_filtered.count()
print("Count of students with Marks > 85:", count)

# First row (Action)
first_row = df_filtered.first()
print("First Row:", first_row)
