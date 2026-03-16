# Ejecuta spark-shell dentro del contenedor con Delta Lake y MinIO (S3A)
# Uso: .\spark-shell-delta.ps1

docker exec -it -u spark spark /opt/spark/bin/spark-shell `
  --packages io.delta:delta-spark_2.12:3.2.0,org.apache.hadoop:hadoop-aws:3.3.4 `
  --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" `
  --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog" `
  --conf "spark.hadoop.fs.s3a.endpoint=http://minio:9000" `
  --conf "spark.hadoop.fs.s3a.access.key=admin" `
  --conf "spark.hadoop.fs.s3a.secret.key=admin123" `
  --conf "spark.hadoop.fs.s3a.path.style.access=true" `
  --conf "spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem"
