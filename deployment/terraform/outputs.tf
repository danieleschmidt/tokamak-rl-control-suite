output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.tokamak_rl.endpoint
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.tokamak_rl.primary_endpoint_address
}

output "data_bucket" {
  description = "S3 bucket for data storage"
  value       = aws_s3_bucket.tokamak_rl_data.id
}

output "models_bucket" {
  description = "S3 bucket for model storage"
  value       = aws_s3_bucket.tokamak_rl_models.id
}
