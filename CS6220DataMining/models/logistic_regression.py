from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
import matplotlib.pyplot as plt


spark = SparkSession.builder \
    .appName("OptimizedLogisticRegression") \
    .config("spark.executor.memory", "6g")\
    .config("spark.driver.memory", "6g")\
    .config("spark.memory.fraction", "0.8")\
    .getOrCreate()


data_path = '../dataset_final.csv'
df = spark.read.csv(data_path, header=True, inferSchema=True)

df = df.withColumn('label', (col('p_recall') > 0.5).cast('int'))


feature_columns = [c for c in df.columns if c not in ['p_recall', 'label']]  # 排除标签列
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")


scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="label", elasticNetParam=0.8, regParam=0.01)


pipeline = Pipeline(stages=[assembler, scaler, lr])


model = pipeline.fit(df)


predictions = model.transform(df)


binary_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
roc_auc = binary_evaluator.evaluate(predictions)

multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
accuracy = multi_evaluator.evaluate(predictions)
precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})

results_path = '../logistic_result_fewer_feature.txt'
with open(results_path, 'w') as file:
    file.write(f"Area Under ROC: {roc_auc}\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"Recall: {recall}\n")
    file.write(f"F1 Score: {f1}\n")

print(f"Results saved to {results_path}")

# Calculate and plot the log likelihood
training_summary = model.stages[-1].summary
objective_history = training_summary.objectiveHistory

plt.figure(figsize=(8, 5))
plt.plot(objective_history)
plt.xlabel('Iteration')
plt.ylabel('Log-Loss')
plt.title('Logistic Regression Log-Loss Curve')
plt.grid(True)
plt.show()

# Save the plot
plt.savefig('log_likelihood_curve.png')
print(f"Log-Loss curve plot saved to /mnt/data/log_likelihood_curve.png")
# 停止 SparkSession
spark.stop()
